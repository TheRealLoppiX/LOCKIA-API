// Log de auditoria do modo Cowork. Cada mensagem (pergunta, veredito do
// classificador de autorização, se foi respondida ou recusada) é registrada
// aqui. Por ora isso é só uma linha JSON estruturada em stdout, capturada
// pelo stream de logs do Render — sem banco novo. Se algum dia precisar
// virar uma tabela de verdade (auditoria pesquisável, retenção definida),
// esta é a única função que precisa mudar.
export interface CoworkAuditEntry {
  userId: string;
  proceed: boolean;
  concern: string;
  messagePreview: string;
}

export function logCoworkEvent(entry: CoworkAuditEntry): void {
  console.log(
    JSON.stringify({
      type: "cowork_audit",
      timestamp: new Date().toISOString(),
      ...entry,
    })
  );
}
