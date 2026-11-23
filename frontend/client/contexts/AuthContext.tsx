import { createContext, useContext, useEffect, useState } from 'react';
import { User, onAuthStateChanged, signOut as firebaseSignOut } from 'firebase/auth';
import { auth } from '@/lib/firebase';

export interface AppUser {
  uid: string;
  email?: string | null;
  displayName?: string | null;
}

interface AuthContextType {
  user: AppUser | null;
  loading: boolean;
  signOut: () => Promise<void>;
  signInAsGuest: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  signOut: async () => {},
  signInAsGuest: () => {},
});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = useState<AppUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if demo guest user was previously saved
    const savedGuest = localStorage.getItem('cellexis_guest_user');
    if (savedGuest) {
      setUser({
        uid: 'nasa-guest-researcher',
        email: 'researcher@cellexis.nasa.gov',
        displayName: 'Guest Researcher'
      });
      setLoading(false);
      return;
    }

    try {
      const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
        if (firebaseUser) {
          setUser(firebaseUser);
        } else if (!localStorage.getItem('cellexis_guest_user')) {
          setUser(null);
        }
        setLoading(false);
      });
      return unsubscribe;
    } catch (e) {
      console.warn('Firebase auth listener bypassed:', e);
      setLoading(false);
    }
  }, []);

  const signInAsGuest = () => {
    const guestUser: AppUser = {
      uid: 'nasa-guest-researcher',
      email: 'researcher@cellexis.nasa.gov',
      displayName: 'Guest Researcher'
    };
    localStorage.setItem('cellexis_guest_user', 'true');
    setUser(guestUser);
    setLoading(false);
  };

  const signOut = async () => {
    try {
      localStorage.removeItem('cellexis_guest_user');
      setUser(null);
      await firebaseSignOut(auth);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <AuthContext.Provider value={{ user, loading, signOut, signInAsGuest }}>
      {children}
    </AuthContext.Provider>
  );
};
