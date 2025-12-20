/**
 * Authentication utilities for LeLamp WebUI
 *
 * Provides Clerk integration with local network bypass support.
 */

import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import {
  ClerkProvider,
  SignIn,
  SignedIn,
  SignedOut,
  useAuth as useClerkAuth,
  useUser,
  UserButton,
} from '@clerk/clerk-react'

// Re-export Clerk components for convenience
export { SignIn, SignedIn, SignedOut, UserButton, useUser }

// =============================================================================
// Types
// =============================================================================

interface AuthConfig {
  enabled: boolean
  localBypass: boolean
  clerkPublishableKey: string | null
}

interface AuthContextValue {
  isAuthenticated: boolean
  isLocalBypass: boolean
  isLoading: boolean
  authEnabled: boolean
  userId: string | null
  getToken: () => Promise<string | null>
}

// =============================================================================
// Auth Context
// =============================================================================

const AuthContext = createContext<AuthContextValue>({
  isAuthenticated: false,
  isLocalBypass: false,
  isLoading: true,
  authEnabled: false,
  userId: null,
  getToken: async () => null,
})

export const useAuth = () => useContext(AuthContext)

// =============================================================================
// Local Network Detection
// =============================================================================

/**
 * Check if we're accessing from a local network.
 * This is a client-side heuristic - the server does the real check.
 */
function isLocalAccess(): boolean {
  const hostname = window.location.hostname

  // Check for localhost
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return true
  }

  // Check for private IP ranges
  const privateRanges = [
    /^10\./,                     // 10.x.x.x
    /^172\.(1[6-9]|2[0-9]|3[0-1])\./, // 172.16.x.x - 172.31.x.x
    /^192\.168\./,               // 192.168.x.x
    /^169\.254\./,               // Link-local
  ]

  return privateRanges.some(range => range.test(hostname))
}

// =============================================================================
// Auth Provider Component
// =============================================================================

interface AuthProviderProps {
  children: ReactNode
  config: AuthConfig
}

/**
 * Inner auth provider that uses Clerk hooks
 */
function ClerkAuthProvider({ children }: { children: ReactNode }) {
  const { isLoaded, isSignedIn, userId, getToken } = useClerkAuth()

  const value: AuthContextValue = {
    isAuthenticated: isSignedIn || false,
    isLocalBypass: false, // Clerk is enabled, so no bypass
    isLoading: !isLoaded,
    authEnabled: true,
    userId: userId || null,
    getToken: async () => {
      const token = await getToken()
      return token
    },
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

/**
 * Auth provider that handles both Clerk and local bypass modes
 */
export function AuthProvider({ children, config }: AuthProviderProps) {
  const [isLocal, setIsLocal] = useState(false)

  useEffect(() => {
    // Check if we're on local network
    setIsLocal(isLocalAccess())
  }, [])

  // If auth is disabled or we're on local network with bypass enabled
  if (!config.enabled || (config.localBypass && isLocal)) {
    const bypassValue: AuthContextValue = {
      isAuthenticated: true,
      isLocalBypass: true,
      isLoading: false,
      authEnabled: config.enabled,
      userId: null,
      getToken: async () => null,
    }

    return (
      <AuthContext.Provider value={bypassValue}>
        {children}
      </AuthContext.Provider>
    )
  }

  // Clerk auth required
  if (!config.clerkPublishableKey) {
    console.error('Clerk publishable key not configured')
    // Allow access but show warning
    const errorValue: AuthContextValue = {
      isAuthenticated: true,
      isLocalBypass: true,
      isLoading: false,
      authEnabled: true,
      userId: null,
      getToken: async () => null,
    }

    return (
      <AuthContext.Provider value={errorValue}>
        {children}
      </AuthContext.Provider>
    )
  }

  return (
    <ClerkProvider publishableKey={config.clerkPublishableKey}>
      <ClerkAuthProvider>
        {children}
      </ClerkAuthProvider>
    </ClerkProvider>
  )
}

// =============================================================================
// Protected Route Component
// =============================================================================

interface ProtectedRouteProps {
  children: ReactNode
  fallback?: ReactNode
}

/**
 * Wrap routes that require authentication
 */
export function ProtectedRoute({ children, fallback }: ProtectedRouteProps) {
  const { isAuthenticated, isLocalBypass, isLoading, authEnabled } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white">Loading...</div>
      </div>
    )
  }

  // Allow access if authenticated or on local bypass
  if (isAuthenticated || isLocalBypass) {
    return <>{children}</>
  }

  // Show sign-in if auth is enabled and not authenticated
  if (authEnabled) {
    return fallback || <SignInPage />
  }

  // Auth disabled, allow access
  return <>{children}</>
}

// =============================================================================
// Sign In Page
// =============================================================================

export function SignInPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="w-full max-w-md p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">LeLamp Dashboard</h1>
          <p className="text-gray-400">Sign in to access your device</p>
        </div>
        <SignIn
          appearance={{
            elements: {
              rootBox: 'mx-auto',
              card: 'bg-gray-800 border-gray-700',
              headerTitle: 'text-white',
              headerSubtitle: 'text-gray-400',
              socialButtonsBlockButton: 'bg-gray-700 border-gray-600 text-white hover:bg-gray-600',
              formFieldLabel: 'text-gray-300',
              formFieldInput: 'bg-gray-700 border-gray-600 text-white',
              footerActionLink: 'text-blue-400 hover:text-blue-300',
            }
          }}
        />
      </div>
    </div>
  )
}

// =============================================================================
// Auth Header Component
// =============================================================================

/**
 * Header component showing auth status
 */
export function AuthHeader() {
  const { isAuthenticated, isLocalBypass, authEnabled } = useAuth()

  if (!authEnabled) {
    return null
  }

  if (isLocalBypass) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span className="w-2 h-2 rounded-full bg-green-500"></span>
        <span>Local Access</span>
      </div>
    )
  }

  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-3">
        <UserButton
          appearance={{
            elements: {
              avatarBox: 'w-8 h-8',
            }
          }}
        />
      </div>
    )
  }

  return null
}
