import "./globals.css";
export const metadata = {
  title: "AI Trading Lab | Research",
  description: "Reproducible market research and realistic backtesting",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
