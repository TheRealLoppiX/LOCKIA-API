import '@fastify/jwt';

// Mesmo shape de payload assinado pelo LOCK-API — o LOCKIA-API nunca assina
// tokens, só verifica os que o LOCK-API já emitiu (mesmo APP_JWT_SECRET).
declare module '@fastify/jwt' {
  interface FastifyJWT {
    payload: {
      sub: string;
      name: string;
      email: string;
      avatar_url?: string;
      is_admin?: boolean;
    };
    user: {
      sub: string;
      name: string;
      email: string;
      avatar_url?: string;
      is_admin?: boolean;
    };
  }
}
