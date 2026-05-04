import type { VercelApiHandler } from "@vercel/node";

export const config = { runtime: "nodejs" } as const;

const handler: VercelApiHandler = async (_req, res) => {
  res.status(200).json({ status: "healthy" });
};

export default handler;
