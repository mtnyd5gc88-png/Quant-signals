import { createContext, useContext, useState, type ReactNode } from 'react';

export interface InvestorProfile {
  riskTolerance: 1 | 2 | 3 | 4 | 5;
  drawdownTolerance: 1 | 2 | 3 | 4;
  timeHorizon: 1 | 2 | 3;
  volatilityPreference: 1 | 2 | 3;
}

export const DEFAULT_PROFILE: InvestorProfile = {
  riskTolerance: 3,
  drawdownTolerance: 2,
  timeHorizon: 2,
  volatilityPreference: 2,
};

const LS_KEY = 'qs_investor_profile';

interface ProfileCtx {
  profile: InvestorProfile;
  setProfile: (p: InvestorProfile) => void;
}

const Ctx = createContext<ProfileCtx>({ profile: DEFAULT_PROFILE, setProfile: () => {} });

export function InvestorProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<InvestorProfile>(() => {
    try {
      const stored = localStorage.getItem(LS_KEY);
      return stored ? (JSON.parse(stored) as InvestorProfile) : DEFAULT_PROFILE;
    } catch { return DEFAULT_PROFILE; }
  });

  const setProfile = (p: InvestorProfile) => {
    localStorage.setItem(LS_KEY, JSON.stringify(p));
    setProfileState(p);
  };

  return <Ctx.Provider value={{ profile, setProfile }}>{children}</Ctx.Provider>;
}

export function useInvestorProfile() {
  return useContext(Ctx);
}

export function isDefaultProfile(p: InvestorProfile): boolean {
  return (
    p.riskTolerance === 3 &&
    p.drawdownTolerance === 2 &&
    p.timeHorizon === 2 &&
    p.volatilityPreference === 2
  );
}
