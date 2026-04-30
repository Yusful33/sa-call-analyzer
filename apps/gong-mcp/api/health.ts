export const config = { runtime: "nodejs" } as const;

export default function handler(): Response {
  return Response.json({ status: "healthy" });
}
