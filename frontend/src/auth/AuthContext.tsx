import { createContext, useContext, useState, type ReactNode } from 'react';

interface User {
  id: string;
  name: string;
  email: string;
  initials: string;
}

interface AuthContextValue {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  getToken: () => string | null;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const token = localStorage.getItem('qs_token');
      const stored = localStorage.getItem('qs_user');
      return token && stored ? (JSON.parse(stored) as User) : null;
    } catch {
      return null;
    }
  });

  const login = async (email: string, password: string) => {
    const baseUrl = localStorage.getItem('qs_api_url') || 'http://localhost:8000';
    const res = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Login failed' }));
      throw new Error((err as { detail?: string }).detail || 'Invalid credentials');
    }
    const data = await res.json() as { access_token: string; user: User };
    localStorage.setItem('qs_token', data.access_token);
    localStorage.setItem('qs_user', JSON.stringify(data.user));
    setUser(data.user);
  };

  const logout = () => {
    localStorage.removeItem('qs_token');
    localStorage.removeItem('qs_user');
    setUser(null);
  };

  const getToken = () => localStorage.getItem('qs_token');

  return (
    <AuthContext.Provider value={{ user, login, logout, getToken }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
