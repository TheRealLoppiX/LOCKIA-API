import 'dotenv/config';
import fastify from "fastify";
import cors from '@fastify/cors';
import jwt from '@fastify/jwt';
import { z } from 'zod';
import { aiService } from './services/AiService.js';
import { logCoworkEvent } from './services/AuditLog.js';

// bodyLimit maior que o padrão (1MB) para caber anexos de imagem/documento
// em base64 no chat — o limite por arquivo é reforçado abaixo.
const app = fastify({ bodyLimit: 20 * 1024 * 1024 });

// ===================================================================
// CONFIGURAÇÃO DOS PLUGINS
// ===================================================================
// Mesmo segredo de assinatura do LOCK-API (APP_JWT_SECRET), copiado
// manualmente nas variáveis de ambiente dos dois serviços — nunca no
// código. O LOCKIA-API nunca assina token nenhum, só verifica os que o
// LOCK-API já emitiu no /login: é assim que o mesmo login funciona nos
// dois produtos sem nenhuma ponte de SSO.
app.register(jwt, { secret: process.env.APP_JWT_SECRET! });

// CORS aqui é mais restrito que no LOCK-API: só a origem do front do
// LOCKIA precisa chamar este serviço direto do navegador. O LOCK-API fala
// com este serviço sempre de servidor pra servidor (sem CORS envolvido).
const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'http://localhost:3002')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

app.register(cors, {
  origin: allowedOrigins,
  methods: ["GET", "POST", "OPTIONS"],
  credentials: true,
  allowedHeaders: ["Content-Type", "Authorization"],
});

// ===================================================================
// ESQUEMAS DE VALIDAÇÃO (ZOD) — espelham os mesmos limites já usados hoje
// no LOCK-API pras rotas de IA, pra manter o mesmo contrato pro cliente.
// ===================================================================
const ALLOWED_ATTACHMENT_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'application/pdf', 'text/plain'];
const MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024; // 5MB por arquivo, antes do base64

const attachmentSchema = z.object({
  name: z.string().min(1).max(200),
  mimeType: z.enum(ALLOWED_ATTACHMENT_TYPES as [string, ...string[]]),
  data: z.string().min(1),
}).refine((att) => Buffer.byteLength(att.data, 'base64') <= MAX_ATTACHMENT_BYTES, {
  message: 'Arquivo maior que o limite de 5MB.',
});

// Histórico da conversa (o cliente manda as trocas anteriores da mesma
// conversa — nada disso é guardado no servidor). Limitado a 40 entradas no
// schema; o serviço ainda corta pra uma janela menor antes de repassar à
// Groq (ver AiService.ts).
const chatHistorySchema = z.array(z.object({
  role: z.enum(['user', 'aegis']),
  content: z.string().max(4000),
})).max(40).optional();

const chatSchema = z.object({
  message: z.string().max(2000).default(''),
  attachments: z.array(attachmentSchema).max(3).optional(),
  history: chatHistorySchema,
}).refine((body) => body.message.trim().length > 0 || (body.attachments && body.attachments.length > 0), {
  message: 'Envie uma mensagem ou pelo menos um anexo.',
});

const challengeSchema = z.object({
  message: z.string().min(1).max(2000),
  history: chatHistorySchema,
});

const coworkSchema = z.object({
  message: z.string().min(1).max(2000),
  history: chatHistorySchema,
  authorizationConfirmed: z.boolean().default(false),
});

const moderateImageSchema = z.object({
  data: z.string().min(1),
  mimeType: z.enum(['image/png', 'image/jpeg']),
});

const generateQuestionsSchema = z.object({
  tema: z.string().min(2).max(100),
  quantidade: z.coerce.number().int().min(1).max(10),
});

// ===================================================================
// ROTA DE HEALTH CHECKING
// ===================================================================
app.get('/ping', async (request, reply) => {
  return reply.send({ message: 'pong' });
});

// ===================================================================
// MODO CHAT — usado tanto pelo LOCK-API (repassando o chat embutido do
// LOCK, depois de sua própria varredura anti-prompt-injection) quanto
// diretamente pelo front do LOCKIA.
// ===================================================================
app.post('/chat', async (request, reply) => {
  try {
    await request.jwtVerify();
    const { message, attachments, history } = chatSchema.parse(request.body);

    // O modelo de visão (usado quando há imagem anexada) "pensa" antes de
    // responder e essa etapa consome tokens da própria resposta — um budget
    // maior evita que a resposta final saia cortada nesses casos.
    const hasImageAttachment = (attachments || []).some((a) => a.mimeType.startsWith('image/'));
    const response = await aiService.askAegis(message, hasImageAttachment ? 1400 : 800, attachments, history);
    return reply.send({ response });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Mensagem inválida.', issues: error.format() });
    }
    console.error('Erro na rota /chat:', error);
    return reply.status(500).send({ message: 'Erro ao processar com IA.' });
  }
});

app.post('/analyze-quiz', async (request, reply) => {
  try {
    await request.jwtVerify();
    const { wrongQuestions } = z.object({
      wrongQuestions: z.array(z.string().min(1)).min(1).max(20),
    }).parse(request.body);

    const analysis = await aiService.analisarErros(wrongQuestions);
    return reply.send({ analysis });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Dados inválidos.' });
    }
    console.error('Erro na rota /analyze-quiz:', error);
    return reply.status(500).send({ message: 'Erro ao analisar resultados.' });
  }
});

// Chamada de servidor para servidor pelo /profile/avatar do LOCK-API — o
// LOCK-API repassa o mesmo Authorization que já recebeu do usuário.
app.post('/moderate-image', async (request, reply) => {
  try {
    await request.jwtVerify();
    const { data, mimeType } = moderateImageSchema.parse(request.body);
    const result = await aiService.moderateImage(data, mimeType);
    return reply.send(result);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Dados inválidos.' });
    }
    console.error('Erro na rota /moderate-image:', error);
    return reply.status(502).send({ message: 'Erro ao moderar imagem.' });
  }
});

// Chamada de servidor para servidor pelo /admin/questions/generate do
// LOCK-API — is_admin já vem no claim do JWT assinado pelo LOCK-API, então
// dá pra conferir aqui sem perguntar nada a mais pra ele.
app.post('/generate-questions', async (request, reply) => {
  try {
    await request.jwtVerify();
    if (!request.user.is_admin) {
      return reply.status(403).send({ message: 'Acesso negado.' });
    }
    const { tema, quantidade } = generateQuestionsSchema.parse(request.body);
    const questoes = await aiService.gerarQuestoesIA(tema, quantidade);
    return reply.send({ questoes });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Dados inválidos.', issues: error.format() });
    }
    console.error('Erro na rota /generate-questions:', error);
    return reply.status(500).send({ message: 'Erro ao gerar questões com IA.' });
  }
});

// ===================================================================
// MODO CHALLENGE — só o front do LOCKIA chama. Gera uma página HTML
// autocontida; o front é responsável por renderizá-la num iframe
// sandboxed (sem allow-same-origin), já que o próprio propósito do modo é
// gerar páginas propositalmente vulneráveis.
// ===================================================================
app.post('/challenge', async (request, reply) => {
  try {
    await request.jwtVerify();
    const { message, history } = challengeSchema.parse(request.body);
    const html = await aiService.generateChallenge(message, history);
    return reply.send({ html });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Mensagem inválida.', issues: error.format() });
    }
    console.error('Erro na rota /challenge:', error);
    return reply.status(500).send({ message: 'Erro ao gerar o desafio.' });
  }
});

// ===================================================================
// MODO COWORK — só o front do LOCKIA chama. Camada de segurança extra:
// gate de consentimento (authorizationConfirmed) + classificador dedicado
// que julga cada mensagem (não só a primeira) + log de auditoria. Ver
// AiService.ts (cowork/judgeCoworkRequest) para o desenho completo.
// ===================================================================
app.post('/cowork', async (request, reply) => {
  try {
    await request.jwtVerify();
    const { message, history, authorizationConfirmed } = coworkSchema.parse(request.body);

    const result = await aiService.cowork(message, history, authorizationConfirmed);

    logCoworkEvent({
      userId: request.user.sub,
      proceed: result.allowed,
      concern: result.concern,
      messagePreview: message.slice(0, 200),
    });

    return reply.send({ response: result.reply });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return reply.status(400).send({ message: 'Mensagem inválida.', issues: error.format() });
    }
    console.error('Erro na rota /cowork:', error);
    return reply.status(500).send({ message: 'Erro ao processar com IA.' });
  }
});

// ===================================================================
// INICIALIZAÇÃO DO SERVIDOR
// ===================================================================
app.listen({
  host: "0.0.0.0",
  port: process.env.PORT ? Number(process.env.PORT) : 3334,
}).then(() => {
  console.log("🤖 LOCKIA-API rodando com CORS ativado em http://localhost:3334");
});
