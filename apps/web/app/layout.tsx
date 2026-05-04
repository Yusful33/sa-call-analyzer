import type { Metadata } from "next";
import "./globals.css";

const metadataBaseUrl =
  process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ||
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "https://stillness.vercel.app");

export const metadata: Metadata = {
  metadataBase: new URL(metadataBaseUrl),
  title: "Stillness",
  description: "360° prospect intelligence — CRM, calls, product usage, and behavior unified.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
