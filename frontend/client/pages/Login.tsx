import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { auth, googleProvider } from '@/lib/firebase';
import { signInWithPopup } from 'firebase/auth';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from "react-router-dom";
import LiquidEther from "@/components/ui/liquid-ether";

export default function Login() {
  const [username, setUsername] = useState("");
  const { signInAsGuest } = useAuth();
  const navigate = useNavigate();

  const signInWithGoogle = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      console.log("User signed in:", user);
      navigate("/dashboard");
    } catch (error) {
      console.error("Error signing in with Google:", error);
    }
  };

  const handleGuestSignIn = () => {
    signInAsGuest();
    navigate("/dashboard");
  };

  return (
    <div className="relative h-screen w-full">
      <div className="absolute inset-0 z-0">
        <LiquidEther
          colors={['#06b6d4', '#10b981', '#0891b2']}
          mouseForce={20}
          cursorSize={100}
          isViscous={false}
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce={false}
          autoSpeed={0.5}
          autoDemo={true}
          autoIntensity={2.2}
          takeoverDuration={0.25}
          autoResumeDelay={3000}
          autoRampDuration={0.6}
        />
      </div>
      <div className="relative z-10 container mx-auto h-full flex items-center justify-center py-16 pointer-events-none">
        <div className="glass glow max-w-md w-full rounded-2xl p-8 pointer-events-auto">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-display text-gradient">
              Welcome to Cellexis
            </h1>
            <p className="mt-2 text-sm text-foreground/70">
              NASA Bioscience AI Research Platform
            </p>
          </div>
          <div className="grid gap-4">
            <Button
              className="w-full h-11 rounded-[10px] bg-gradient-to-r from-primary via-accent to-secondary text-black font-semibold shadow-lg hover:opacity-95 transition"
              onClick={handleGuestSignIn}
            >
              Explore as Guest Researcher
            </Button>
            <div className="relative flex py-1 items-center">
              <div className="flex-grow border-t border-border/40"></div>
              <span className="flex-shrink mx-4 text-xs text-foreground/50 uppercase tracking-widest">or</span>
              <div className="flex-grow border-t border-border/40"></div>
            </div>
            <Button
              variant="outline"
              className="w-full h-11 rounded-[10px] border-primary/30 hover:bg-primary/10 text-foreground font-medium"
              onClick={signInWithGoogle}
            >
              Sign in with Google
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
