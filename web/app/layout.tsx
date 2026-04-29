import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
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
