export const config = { runtime: "nodejs" } as const;

export default {
  fetch(): Response {
    return Response.json({ status: "healthy" });
  },
};
