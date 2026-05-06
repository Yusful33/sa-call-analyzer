import type { Metadata } from "next";
import { Rubik } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { ToastProvider, ToastContainer } from "@/components/Toast";
import "./globals.css";

const rubik = Rubik({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

const metadataBaseUrl =
  process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ||
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "https://arize-gtm-stillness.vercel.app");

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
    <html lang="en" className={rubik.variable}>
      <body>
        <ToastProvider>
          {children}
          <ToastContainer />
        </ToastProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
