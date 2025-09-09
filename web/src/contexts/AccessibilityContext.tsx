import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface AccessibilitySettings {
  highContrast: boolean;
  reducedMotion: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'xl';
  focusVisible: boolean;
  announcements: boolean;
  screenReaderOptimized: boolean;
  keyboardNavigation: boolean;
}

interface AccessibilityContextType {
  settings: AccessibilitySettings;
  updateSetting: <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => void;
  announceToScreenReader: (message: string, priority?: 'polite' | 'assertive') => void;
  // Convenience getters
  highContrast: boolean;
  reducedMotion: boolean;
  fontSize: string;
  focusVisible: boolean;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

const STORAGE_KEY = 'dps-accessibility-settings';

const defaultSettings: AccessibilitySettings = {
  highContrast: false,
  reducedMotion: false,
  fontSize: 'medium',
  focusVisible: true,
  announcements: true,
  screenReaderOptimized: false,
  keyboardNavigation: true,
};

interface AccessibilityProviderProps {
  children: ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [settings, setSettings] = useState<AccessibilitySettings>(defaultSettings);
  const [announceElement, setAnnounceElement] = useState<HTMLElement | null>(null);

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsedSettings = JSON.parse(saved);
        setSettings({ ...defaultSettings, ...parsedSettings });
      }
    } catch (error) {
      console.warn('Failed to load accessibility settings:', error);
    }

    // Detect system preferences
    detectSystemPreferences();
    
    // Create announcement element for screen readers
    createAnnouncementElement();
  }, []);

  // Save settings to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.warn('Failed to save accessibility settings:', error);
    }

    // Apply settings to document
    applySettingsToDocument();
  }, [settings]);

  const detectSystemPreferences = () => {
    // Detect reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setSettings(prev => ({ ...prev, reducedMotion: true }));
    }

    // Detect high contrast preference
    if (window.matchMedia('(prefers-contrast: high)').matches) {
      setSettings(prev => ({ ...prev, highContrast: true }));
    }

    // Listen for changes
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');

    motionQuery.addEventListener('change', (e) => {
      setSettings(prev => ({ ...prev, reducedMotion: e.matches }));
    });

    contrastQuery.addEventListener('change', (e) => {
      setSettings(prev => ({ ...prev, highContrast: e.matches }));
    });
  };

  const createAnnouncementElement = () => {
    // Create live region for screen reader announcements
    const element = document.createElement('div');
    element.id = 'accessibility-announcements';
    element.setAttribute('aria-live', 'polite');
    element.setAttribute('aria-atomic', 'true');
    element.style.position = 'absolute';
    element.style.left = '-10000px';
    element.style.width = '1px';
    element.style.height = '1px';
    element.style.overflow = 'hidden';
    
    document.body.appendChild(element);
    setAnnounceElement(element);
  };

  const applySettingsToDocument = () => {
    const root = document.documentElement;
    
    // High contrast
    if (settings.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Reduced motion
    if (settings.reducedMotion) {
      root.classList.add('reduced-motion');
    } else {
      root.classList.remove('reduced-motion');
    }

    // Font size
    root.classList.remove('text-small', 'text-medium', 'text-large', 'text-xl');
    root.classList.add(`text-${settings.fontSize}`);

    // Focus visible
    if (settings.focusVisible) {
      root.classList.add('focus-visible-enabled');
    } else {
      root.classList.remove('focus-visible-enabled');
    }

    // Screen reader optimizations
    if (settings.screenReaderOptimized) {
      root.classList.add('screen-reader-optimized');
    } else {
      root.classList.remove('screen-reader-optimized');
    }
  };

  const updateSetting = <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const announceToScreenReader = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!settings.announcements || !announceElement) return;

    // Update aria-live attribute based on priority
    announceElement.setAttribute('aria-live', priority);
    
    // Clear and then set the message to ensure it's announced
    announceElement.textContent = '';
    
    // Use setTimeout to ensure the clear happens first
    setTimeout(() => {
      announceElement.textContent = message;
    }, 100);

    // Clear the message after announcement
    setTimeout(() => {
      announceElement.textContent = '';
      announceElement.setAttribute('aria-live', 'polite'); // Reset to polite
    }, 2000);
  };

  const value: AccessibilityContextType = {
    settings,
    updateSetting,
    announceToScreenReader,
    // Convenience getters
    highContrast: settings.highContrast,
    reducedMotion: settings.reducedMotion,
    fontSize: settings.fontSize,
    focusVisible: settings.focusVisible,
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
    </AccessibilityContext.Provider>
  );
};

export const useAccessibility = (): AccessibilityContextType => {
  const context = useContext(AccessibilityContext);
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};

// Custom hook for keyboard navigation
export const useKeyboardNavigation = () => {
  const { settings } = useAccessibility();
  
  useEffect(() => {
    if (!settings.keyboardNavigation) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Skip links on Tab
      if (event.key === 'Tab' && !event.shiftKey) {
        const skipLinks = document.querySelectorAll('[data-skip-link]');
        if (skipLinks.length > 0 && document.activeElement === skipLinks[0]) {
          event.preventDefault();
          const target = document.querySelector('#main-content');
          if (target instanceof HTMLElement) {
            target.focus();
          }
        }
      }
      
      // Escape key handling
      if (event.key === 'Escape') {
        // Close modals, dropdowns, etc.
        const closeButtons = document.querySelectorAll('[data-close-on-escape]');
        closeButtons.forEach(button => {
          if (button instanceof HTMLElement) {
            button.click();
          }
        });
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [settings.keyboardNavigation]);
};

// Hook for focus management
export const useFocusManagement = () => {
  const { settings } = useAccessibility();
  
  const focusElement = (selector: string) => {
    if (!settings.focusVisible) return;
    
    const element = document.querySelector(selector);
    if (element instanceof HTMLElement) {
      element.focus();
    }
  };
  
  const trapFocus = (containerSelector: string) => {
    const container = document.querySelector(containerSelector);
    if (!container) return;
    
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    const handleTab = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      
      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };
    
    container.addEventListener('keydown', handleTab);
    
    // Return cleanup function
    return () => container.removeEventListener('keydown', handleTab);
  };
  
  return { focusElement, trapFocus };
};