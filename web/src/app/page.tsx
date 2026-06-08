import { AuthProvider } from "@/components/auth-provider";
import { BrainApp } from "@/components/brain-app";

export default function Home() {
  return (
    <AuthProvider>
      <BrainApp />
    </AuthProvider>
  );
}
