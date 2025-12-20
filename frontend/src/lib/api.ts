/**
 * LeLamp API client
 *
 * Provides type-safe API calls to the LeLamp backend.
 */

const API_BASE = '/api/v1'

// Response type for API calls (used for type reference)
type ApiResponse<T> = {
  success: boolean
  error?: string
  data?: T
}
export type { ApiResponse }

// Auth token getter - set by AuthProvider
let getAuthToken: (() => Promise<string | null>) | null = null

/**
 * Set the auth token getter function.
 * Called by AuthProvider to enable authenticated API calls.
 */
export function setAuthTokenGetter(getter: () => Promise<string | null>) {
  getAuthToken = getter
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  // Build headers with optional auth token
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string>),
  }

  // Add auth token if available
  if (getAuthToken) {
    const token = await getAuthToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error('Authentication required')
    }
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

// Calibration API
export const calibrationApi = {
  start: () =>
    fetchApi<{ success: boolean; error?: string }>('/setup/calibration/start', {
      method: 'POST',
    }),

  getStatus: () =>
    fetchApi<{
      connected: boolean
      step: string
      calibration_required: boolean
      range_mins: Record<string, number>
      range_maxs: Record<string, number>
      homing_offsets: Record<string, number>
    }>('/setup/calibration/status'),

  getPositions: () =>
    fetchApi<{
      success: boolean
      positions: Record<string, number>
      step?: string
      range_mins?: Record<string, number>
      range_maxs?: Record<string, number>
    }>('/setup/calibration/positions'),

  recordHoming: () =>
    fetchApi<{ success: boolean; error?: string }>('/setup/calibration/record-homing', {
      method: 'POST',
    }),

  startRange: () =>
    fetchApi<{ success: boolean; error?: string }>('/setup/calibration/start-range', {
      method: 'POST',
    }),

  recordRanges: () =>
    fetchApi<{ success: boolean; error?: string }>('/setup/calibration/record-ranges', {
      method: 'POST',
    }),

  finalize: () =>
    fetchApi<{ success: boolean; calibration_path?: string; error?: string }>(
      '/setup/calibration/finalize',
      { method: 'POST' }
    ),

  cancel: () =>
    fetchApi<{ success: boolean }>('/setup/calibration/cancel', {
      method: 'POST',
    }),
}

// Setup API
export const setupApi = {
  getStatus: () => fetchApi<{
    success: boolean
    first_boot: boolean
    setup_complete: boolean
    current_step: string
    steps_completed: Record<string, boolean>
  }>('/setup/status'),

  updateStep: (step: string) =>
    fetchApi('/setup/step/', {
      method: 'POST',
      body: JSON.stringify({ step }),
    }),

  completeStep: (step: string) =>
    fetchApi('/setup/complete-step/', {
      method: 'POST',
      body: JSON.stringify({ step }),
    }),

  // Skip a setup step and optionally disable a feature
  skipStep: (step: string, disableFeature?: string) =>
    fetchApi<{ success: boolean; message?: string; error?: string }>('/setup/skip-step/', {
      method: 'POST',
      body: JSON.stringify({ step, disable_feature: disableFeature }),
    }),

  finish: () =>
    fetchApi('/setup/finish', { method: 'POST' }),

  restart: () =>
    fetchApi('/setup/restart', { method: 'POST' }),

  // Environment
  checkEnv: () =>
    fetchApi<{
      success: boolean
      exists: boolean
      has_openai: boolean
      has_livekit: boolean
    }>('/setup/env/check'),

  saveEnv: (config: {
    openai_key: string
    livekit_url?: string
    livekit_key?: string
    livekit_secret?: string
  }) =>
    fetchApi('/setup/env/save/', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // Personality
  getPersonality: () =>
    fetchApi<{
      success: boolean
      name: string
      character_id: string
      character: {
        id: string
        name: string
        description: string
        speech_style: string
        voice_model: string
        visual_description?: string
        ideals?: string
        flaws?: string
        bio?: string
      } | null
      default_color?: number[]
      characters: Array<{
        id: string
        name: string
        description: string
        speech_style: string
        voice_model: string
      }>
    }>('/setup/personality/'),

  savePersonality: (config: {
    name: string
    character_id: string
    default_color?: number[]
  }) =>
    fetchApi('/setup/personality/', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // Location
  getLocation: () =>
    fetchApi<{
      success: boolean
      city: string
      region: string
      country: string
      timezone: string
      lat: number
      lon: number
    }>('/setup/location/'),

  saveLocation: (config: {
    city: string
    region?: string
    country?: string
    lat: number
    lon: number
  }) =>
    fetchApi('/setup/location/', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // AI Backend
  getAIBackendOptions: () =>
    fetchApi<{
      success: boolean
      backends: Array<{
        id: string
        name: string
        description: string
        requires: string[]
        features: string[]
        recommended: boolean
        coming_soon: boolean
        configured: boolean
      }>
      providers: Array<{
        id: string
        name: string
        description: string
        api_key_env: string
        api_key_url: string
        recommended: boolean
        coming_soon: boolean
        configured: boolean
      }>
      current: string
      current_provider: string
    }>('/setup/ai-backend/options'),

  getAIBackendCurrent: () =>
    fetchApi<{
      success: boolean
      backend: string
      provider?: string
      name: string
      voice: string
      configured: boolean
      missing_keys: string[]
    }>('/setup/ai-backend/current'),

  getVoices: (backend: string, provider?: string) =>
    fetchApi<{
      success: boolean
      backend: string
      provider?: string
      voices: Array<{
        id: string
        name: string
        description: string
        gender: string
        default?: boolean
      }>
      current: string
    }>(`/setup/ai-backend/voices/${backend}${provider ? `?provider=${provider}` : ''}`),

  validateOpenAIKey: (key: string) =>
    fetchApi<{
      success: boolean
      valid: boolean
      error?: string
    }>('/setup/ai-backend/validate-key', {
      method: 'POST',
      body: JSON.stringify({ key }),
    }),

  // Returns URL for voice test audio (for local backend)
  getVoiceTestUrl: (backend: string, voiceId: string, text?: string) =>
    `${API_BASE}/setup/ai-backend/test-voice?backend=${backend}&voice_id=${encodeURIComponent(voiceId)}&text=${encodeURIComponent(text || 'Hello! I am your LeLamp, ready to assist you.')}`,

  // POST for voice test (returns audio file for local, info for livekit)
  testVoice: (backend: string, voiceId: string, text?: string) =>
    fetchApi<{
      success: boolean
      voice?: { id: string; name: string; description: string; gender: string }
      message?: string
      error?: string
    }>('/setup/ai-backend/test-voice', {
      method: 'POST',
      body: JSON.stringify({
        backend,
        voice_id: voiceId,
        text: text || 'Hello! I am your LeLamp, ready to assist you.',
      }),
    }),

  getLocalConfig: () =>
    fetchApi<{
      success: boolean
      ollama: {
        available: boolean
        url: string
        models: string[]
        current_model: string
      }
      whisper: {
        models: string[]
        current_model: string
      }
      piper: {
        available: boolean
        voices: Array<{
          id: string
          name: string
          description: string
          gender: string
          default?: boolean
        }>
        current_voice: string
      }
    }>('/setup/ai-backend/local-config'),

  configureAIBackend: (config: {
    backend: string
    provider?: string
    api_key?: string
    voice?: string
    // Legacy fields
    openai_key?: string
    openai_voice?: string
    // Local pipeline
    ollama_url?: string
    ollama_model?: string
    whisper_model?: string
    piper_voice?: string
  }) =>
    fetchApi<{
      success: boolean
      backend?: string
      provider?: string
      message?: string
      error?: string
      restart_required?: boolean
    }>('/setup/ai-backend/configure/', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  // LiveKit Cloud
  getLiveKitStatus: () =>
    fetchApi<{
      success: boolean
      configured: boolean
      room_name: string
      service_status?: string
      connected?: boolean
      missing_keys?: string[]
      openai_voice?: string
      url: string
      api_key: string
      api_secret_masked: string
    }>('/setup/livekit/status'),

  getLiveKitGuide: () =>
    fetchApi<{
      success: boolean
      steps: Array<{
        step: number
        title: string
        description: string
        url?: string
      }>
      room_name: string
      free_tier_info: string
    }>('/setup/livekit/guide'),

  configureLiveKit: (config: {
    url: string
    api_key: string
    api_secret: string
  }) =>
    fetchApi<{
      success: boolean
      message?: string
      room_name?: string
      error?: string
      restart_required?: boolean
    }>('/setup/livekit/configure', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  testLiveKitConnection: () =>
    fetchApi<{
      success: boolean
      message: string
      room_name?: string
      error?: string
    }>('/setup/livekit/test', {
      method: 'POST',
    }),

  // WiFi
  getWifiStatus: () =>
    fetchApi<{
      success: boolean
      connected: boolean
      has_internet: boolean
      ssid?: string
      mode: 'station' | 'ap' | 'disconnected'
      local_ip?: string
      wan_ip?: string
      latency_ms?: number
    }>('/setup/wifi/status'),

  scanWifi: () =>
    fetchApi<{
      success: boolean
      networks: Array<{
        ssid: string
        signal: number
        security: string
        connected: boolean
      }>
      error?: string
    }>('/setup/wifi/scan'),

  connectWifi: (ssid: string, password: string) =>
    fetchApi<{
      success: boolean
      ssid?: string
      ip_address?: string
      message?: string
      error?: string
    }>('/setup/wifi/connect', {
      method: 'POST',
      body: JSON.stringify({ ssid, password }),
    }),

  skipWifi: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/wifi/skip', {
      method: 'POST',
    }),

  // ==========================================================================
  // Audio Setup
  // ==========================================================================

  getAudioStatus: () =>
    fetchApi<{
      success: boolean
      available: boolean
      has_speaker: boolean
      has_microphone: boolean
      has_usb_audio: boolean
    }>('/setup/audio/status'),

  getAudioDevices: () =>
    fetchApi<{
      success: boolean
      playback: Array<{ name: string; card_index: number; device_type: string }>
      capture: Array<{ name: string; card_index: number; device_type: string }>
    }>('/setup/audio/devices'),

  testSpeaker: () =>
    fetchApi<{ success: boolean; playing?: boolean; message?: string; error?: string }>(
      '/setup/audio/test-speaker',
      { method: 'POST' }
    ),

  stopSpeakerTest: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/audio/test-speaker/stop', {
      method: 'POST',
    }),

  getSpeakerTestStatus: () =>
    fetchApi<{ success: boolean; playing: boolean }>('/setup/audio/test-speaker/status'),

  testMicrophone: () =>
    fetchApi<{ success: boolean; message?: string; error?: string }>(
      '/setup/audio/test-mic',
      { method: 'POST' }
    ),

  startMicMonitoring: () =>
    fetchApi<{ success: boolean; monitoring: boolean; message?: string }>(
      '/setup/audio/monitor/start',
      { method: 'POST' }
    ),

  stopMicMonitoring: () =>
    fetchApi<{ success: boolean; monitoring: boolean; message?: string }>(
      '/setup/audio/monitor/stop',
      { method: 'POST' }
    ),

  getMicMonitoringStatus: () =>
    fetchApi<{ success: boolean; monitoring: boolean }>('/setup/audio/monitor/status'),

  getMicLevel: () =>
    fetchApi<{ success: boolean; level: number }>('/setup/audio/mic-level'),

  getAudioVolumes: () =>
    fetchApi<{
      success: boolean
      speaker_volume: number
      microphone_volume: number
    }>('/setup/audio/volume'),

  setAudioVolumes: (speakerVolume?: number, micVolume?: number) =>
    fetchApi<{
      success: boolean
      results: {
        speaker?: { success: boolean; volume: number }
        microphone?: { success: boolean; volume: number }
      }
    }>('/setup/audio/volume', {
      method: 'POST',
      body: JSON.stringify({
        speaker_volume: speakerVolume,
        microphone_volume: micVolume,
      }),
    }),

  skipAudioSetup: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/audio/skip', {
      method: 'POST',
    }),

  completeAudioSetup: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/audio/complete', {
      method: 'POST',
    }),

  // Mic Calibration
  getMicPresets: () =>
    fetchApi<{
      success: boolean
      presets: Record<string, { target_low: number; target_high: number; description: string }>
      current: string
    }>('/setup/audio/calibration/presets'),

  startMicCalibration: (preset: string = 'normal') =>
    fetchApi<{
      success: boolean
      message?: string
      preset: string
      target_range: [number, number]
      error?: string
    }>(`/setup/audio/calibration/start?preset=${preset}`, { method: 'POST' }),

  submitCalibrationSample: (level: number) =>
    fetchApi<{
      success: boolean
      status: string
      action: 'wait' | 'increase' | 'decrease' | 'complete'
      volume: number
      avg_level?: number
      peak_level?: number
      adjustments?: number
      error?: string
    }>(`/setup/audio/calibration/sample?level=${level}`, { method: 'POST' }),

  stopMicCalibration: () =>
    fetchApi<{ success: boolean; message?: string; final_volume: number }>(
      '/setup/audio/calibration/stop',
      { method: 'POST' }
    ),

  getCalibrationStatus: () =>
    fetchApi<{
      success: boolean
      active: boolean
      status: string
      volume: number
      adjustments: number
      preset: string
    }>('/setup/audio/calibration/status'),

  // ==========================================================================
  // Camera Setup
  // ==========================================================================

  getCameraStatus: () =>
    fetchApi<{
      success: boolean
      available: boolean
      camera_count: number
      working_count: number
    }>('/setup/camera/status'),

  getCameraDevices: () =>
    fetchApi<{
      success: boolean
      cameras: Array<{
        path: string
        actual_device: string
        name: string
        type: string
        has_mic: boolean
        working: boolean
      }>
    }>('/setup/camera/devices'),

  getCameraPreviewUrl: (device: string) =>
    `${API_BASE}/setup/camera/preview?device=${encodeURIComponent(device)}`,

  selectCamera: (device: string, cameraType?: string) =>
    fetchApi<{
      success: boolean
      device?: string
      camera_type?: string
      message?: string
      error?: string
    }>('/setup/camera/select', {
      method: 'POST',
      body: JSON.stringify({ device, camera_type: cameraType || 'auto' }),
    }),

  skipCameraSetup: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/camera/skip', {
      method: 'POST',
    }),

  // ==========================================================================
  // RGB Setup
  // ==========================================================================

  getRgbStatus: () =>
    fetchApi<{
      success: boolean
      available: boolean
      enabled: boolean
      led_count: number
      brightness: number
    }>('/setup/rgb/status'),

  testRgb: () =>
    fetchApi<{ success: boolean; message?: string; error?: string; testing?: boolean }>(
      '/setup/rgb/test',
      { method: 'POST' }
    ),

  setRgbColor: (r: number, g: number, b: number) =>
    fetchApi<{ success: boolean; color?: { r: number; g: number; b: number } }>(
      '/setup/rgb/color',
      {
        method: 'POST',
        body: JSON.stringify({ r, g, b }),
      }
    ),

  turnOffRgb: () =>
    fetchApi<{ success: boolean; message?: string }>('/setup/rgb/off', {
      method: 'POST',
    }),

  getRgbBrightness: () =>
    fetchApi<{ success: boolean; brightness: number }>('/setup/rgb/brightness'),

  setRgbBrightness: (brightness: number) =>
    fetchApi<{ success: boolean; brightness: number; message?: string }>(
      '/setup/rgb/brightness',
      {
        method: 'POST',
        body: JSON.stringify({ brightness }),
      }
    ),

  skipRgbSetup: () =>
    fetchApi<{ success: boolean; message?: string; error?: string }>('/setup/rgb/skip', {
      method: 'POST',
    }),

  completeRgbSetup: () =>
    fetchApi<{ success: boolean; message?: string; error?: string }>('/setup/rgb/complete', {
      method: 'POST',
    }),
}

// Dashboard API
export const dashboardApi = {
  getStatus: () =>
    fetchApi<{
      success: boolean
      agent: { running: boolean; sleeping: boolean }
      services: Record<string, boolean>
      config: { name: string; setup_complete: boolean; calibration_required: boolean }
    }>('/dashboard/status'),

  getSettings: () =>
    fetchApi<{ success: boolean; config: Record<string, unknown> }>(
      '/dashboard/settings/'
    ),

  updateSettings: (settings: Record<string, unknown>) =>
    fetchApi('/dashboard/settings/', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
}

// Agent API
export const agentApi = {
  getStatus: () =>
    fetchApi<{ success: boolean; running: boolean; sleeping: boolean }>(
      '/agent/status'
    ),

  getEnabled: () =>
    fetchApi<{ success: boolean; enabled: boolean }>('/agent/enabled'),

  enable: () => fetchApi('/agent/enable', { method: 'POST' }),

  disable: () => fetchApi('/agent/disable', { method: 'POST' }),

  wake: () => fetchApi('/agent/wake', { method: 'POST' }),

  sleep: () => fetchApi('/agent/sleep', { method: 'POST' }),

  shutdown: () => fetchApi('/agent/shutdown', { method: 'POST' }),

  restartService: () => fetchApi('/agent/restart-service', { method: 'POST' }),

  reboot: () => fetchApi('/agent/reboot', { method: 'POST' }),

  poweroff: () => fetchApi('/agent/poweroff', { method: 'POST' }),
}

// Motors API
export const motorsApi = {
  getMotors: () =>
    fetchApi<Record<string, { id: number; min: number; max: number; current: number }>>(
      '/dashboard/motors/'
    ),

  getPositions: () =>
    fetchApi<Record<string, number>>('/dashboard/motors/positions'),

  move: (motor: string, position: number) =>
    fetchApi('/dashboard/motors/move/', {
      method: 'POST',
      body: JSON.stringify({ motor, position }),
    }),

  setManualControl: (enabled: boolean) =>
    fetchApi('/dashboard/motors/manual-control/', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),

  release: () => fetchApi('/dashboard/motors/release/', { method: 'POST' }),

  setPushableMode: (enabled: boolean) =>
    fetchApi('/dashboard/motors/pushable-mode/', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),

  getStatus: () =>
    fetchApi<{
      success: boolean
      connected: boolean
      manual_control: boolean
      pushable_mode: boolean
    }>('/dashboard/motors/status'),
}

// Tracking API
export const trackingApi = {
  getStatus: () =>
    fetchApi<{ enabled: boolean; available: boolean }>('/dashboard/tracking/status'),

  enable: () => fetchApi('/dashboard/tracking/enable/', { method: 'POST' }),

  disable: () => fetchApi('/dashboard/tracking/disable/', { method: 'POST' }),

  getConfig: () => fetchApi('/dashboard/tracking/config/'),

  updateConfig: (config: Record<string, unknown>) =>
    fetchApi('/dashboard/tracking/config/', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
}

// Services API
export const servicesApi = {
  getStatus: () =>
    fetchApi<{
      success: boolean
      services: Record<string, { enabled: boolean; config_path: string }>
    }>('/dashboard/services/'),

  toggle: (service: string, enabled: boolean) =>
    fetchApi<{ success: boolean; service: string; enabled: boolean; message: string }>(
      '/dashboard/services/toggle/',
      {
        method: 'POST',
        body: JSON.stringify({ service, enabled }),
      }
    ),

  enable: (service: string) =>
    fetchApi<{ success: boolean; service: string; enabled: boolean; message: string }>(
      `/dashboard/services/enable/${service}`,
      { method: 'POST' }
    ),

  disable: (service: string) =>
    fetchApi<{ success: boolean; service: string; enabled: boolean; message: string }>(
      `/dashboard/services/disable/${service}`,
      { method: 'POST' }
    ),

  // RGB brightness controls
  getRgbBrightness: () =>
    fetchApi<{ success: boolean; brightness: number; source: string }>(
      '/dashboard/services/rgb/brightness'
    ),

  setRgbBrightness: (brightness: number) =>
    fetchApi<{ success: boolean; brightness: number; message: string; applied: boolean }>(
      '/dashboard/services/rgb/brightness/',
      {
        method: 'POST',
        body: JSON.stringify({ brightness }),
      }
    ),
}

// Health check
export const healthCheck = () => fetchApi<{ status: string; version: string }>('/health')

// Animations API
export const animationsApi = {
  list: () =>
    fetchApi<{
      success: boolean
      animations: Array<{
        name: string
        frames: number
        duration: number
        size_kb: number
      }>
      current: string | null
      recording_active: boolean
    }>('/dashboard/animations/'),

  play: (name: string) =>
    fetchApi<{ success: boolean; name: string; message: string; error?: string }>(
      '/dashboard/animations/play/',
      {
        method: 'POST',
        body: JSON.stringify({ name }),
      }
    ),

  prepareRecord: () =>
    fetchApi<{ success: boolean; message: string; error?: string }>(
      '/dashboard/animations/prepare-record/',
      { method: 'POST' }
    ),

  startRecord: (name: string) =>
    fetchApi<{ success: boolean; name: string; message: string; error?: string }>(
      '/dashboard/animations/start-record/',
      {
        method: 'POST',
        body: JSON.stringify({ name }),
      }
    ),

  stopRecord: () =>
    fetchApi<{
      success: boolean
      name: string
      frames: number
      duration: number
      message: string
      error?: string
    }>('/dashboard/animations/stop-record/', { method: 'POST' }),

  cancelRecord: () =>
    fetchApi<{ success: boolean; message: string; error?: string }>(
      '/dashboard/animations/cancel-record/',
      { method: 'POST' }
    ),

  getRecordingStatus: () =>
    fetchApi<{
      recording: boolean
      name: string | null
      frames: number
      duration: number
    }>('/dashboard/animations/recording-status'),

  delete: (name: string) =>
    fetchApi<{ success: boolean; name: string; message: string; error?: string }>(
      `/dashboard/animations/${name}`,
      { method: 'DELETE' }
    ),
}

// Theme API
export const themeApi = {
  getThemes: () =>
    fetchApi<{
      success: boolean
      current: string
      themes: Array<{ name: string; sound_count: number; is_current: boolean }>
    }>('/dashboard/theme/'),

  setTheme: (name: string) =>
    fetchApi<{ success: boolean; theme: string; message: string; error?: string }>(
      '/dashboard/theme/',
      {
        method: 'POST',
        body: JSON.stringify({ name }),
      }
    ),

  getThemeInfo: (name: string) =>
    fetchApi<{
      success: boolean
      name: string
      path: string
      sounds: Array<{ name: string; exists: boolean }>
    }>(`/dashboard/theme/${name}`),
}

// Dance Mode API
export const danceApi = {
  getStatus: () =>
    fetchApi<{
      success: boolean
      dance_mode?: boolean
      dance_threshold?: number
      excited_threshold?: number
      current_energy?: number
      error?: string
    }>('/modifiers/dance/'),

  enable: () =>
    fetchApi<{ success: boolean; message?: string; error?: string }>(
      '/modifiers/dance/enable/',
      { method: 'POST' }
    ),

  disable: () =>
    fetchApi<{ success: boolean; message?: string; error?: string }>(
      '/modifiers/dance/disable/',
      { method: 'POST' }
    ),
}

// Music Modifier API (real-time beat sync status)
export const musicModifierApi = {
  getStatus: () =>
    fetchApi<{
      enabled: boolean
      amplitude?: number
      beat_divisor?: number
      groove?: number
      active_joints?: string[]
      available_joints?: string[]
      fallback_bpm?: number
      current_bpm?: number
      error?: string
    }>('/modifiers/music/'),

  enable: () =>
    fetchApi<{ success: boolean; error?: string }>(
      '/modifiers/music/enable',
      { method: 'POST' }
    ),

  disable: () =>
    fetchApi<{ success: boolean; error?: string }>(
      '/modifiers/music/disable',
      { method: 'POST' }
    ),
}

// Workflows API
export const workflowsApi = {
  list: () =>
    fetchApi<{
      workflows: Array<{
        id: string
        name: string
        description: string
        author: string
        node_count: number
        edge_count: number
      }>
    }>('/workflows/'),

  get: (workflowId: string) =>
    fetchApi<{
      id: string
      name: string
      description: string
      author: string
      state_schema: Record<string, any>
      nodes: Array<{
        id: string
        intent: string
        preferred_actions: string[]
        type: string
        position: { x: number; y: number }
      }>
      edges: Array<{
        id: string
        source: string
        target: string | Record<string, string>
        type: string
        state_key?: string
        comment?: string
      }>
    }>(`/workflows/${workflowId}`),
}

// Spotify API
const SPOTIFY_API_BASE = '/api/v1/spotify'

async function fetchSpotifyApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${SPOTIFY_API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`Spotify API error: ${response.status}`)
  }

  return response.json()
}

export const spotifyApi = {
  // Status & Auth
  getStatus: () =>
    fetchSpotifyApi<{
      enabled: boolean
      authenticated: boolean
      is_playing?: boolean
      current_track?: any
      device_name?: string
      message?: string
    }>('/status'),

  getCallbackUrl: () =>
    fetchSpotifyApi<{ success: boolean; ip: string; callback_url: string }>('/callback-url'),

  getAuthUrl: () =>
    fetchSpotifyApi<{ success: boolean; auth_url?: string; error?: string }>('/auth/url'),

  submitAuthCode: (code: string) =>
    fetchSpotifyApi<{ success: boolean; message?: string; error?: string }>('/auth/code/', {
      method: 'POST',
      body: JSON.stringify({ code }),
    }),

  saveCredentials: (clientId: string, clientSecret: string) =>
    fetchSpotifyApi<{ success: boolean; message?: string; error?: string }>('/credentials/', {
      method: 'POST',
      body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
    }),

  updateDeviceName: (deviceName: string) =>
    fetchSpotifyApi<{ success: boolean; message?: string; error?: string }>('/device-name/', {
      method: 'POST',
      body: JSON.stringify({ device_name: deviceName }),
    }),

  // Playback
  getCurrentTrack: () =>
    fetchSpotifyApi<{
      success: boolean
      is_playing?: boolean
      track_name?: string
      artist?: string
      album?: string
      album_art?: string
      progress_ms?: number
      duration_ms?: number
      volume?: number
      shuffle?: boolean
      repeat?: string
      error?: string
    }>('/current'),

  play: () => fetchSpotifyApi<{ success: boolean; error?: string }>('/play', { method: 'POST' }),

  pause: () => fetchSpotifyApi<{ success: boolean; error?: string }>('/pause', { method: 'POST' }),

  next: () => fetchSpotifyApi<{ success: boolean; error?: string }>('/next', { method: 'POST' }),

  previous: () => fetchSpotifyApi<{ success: boolean; error?: string }>('/previous', { method: 'POST' }),

  setVolume: (volume: number) =>
    fetchSpotifyApi<{ success: boolean; error?: string }>('/volume/', {
      method: 'POST',
      body: JSON.stringify({ volume }),
    }),

  setShuffle: (state: boolean) =>
    fetchSpotifyApi<{ success: boolean; error?: string }>('/shuffle/', {
      method: 'POST',
      body: JSON.stringify({ state }),
    }),

  setRepeat: (state: string) =>
    fetchSpotifyApi<{ success: boolean; error?: string }>('/repeat/', {
      method: 'POST',
      body: JSON.stringify({ state }),
    }),
}

// System API (device info, fan control, etc.)
export const systemApi = {
  getInfo: () =>
    fetchApi<{
      success: boolean
      connected: boolean
      device_name: string
      temperature: number | null
      cpu_percent: number | null
      memory: {
        total_mb: number
        used_mb: number
        free_mb: number
        available_mb: number
        percent: number
      }
      disk: {
        total_gb: number
        used_gb: number
        free_gb: number
        percent: number
      }
      uptime: {
        seconds: number
        formatted: string
      }
      network: {
        wifi_status: {
          connected: boolean
          ssid: string | null
          mode: string
          interface: string
        }
        internet_status: {
          connected: boolean
          latency_ms: number | null
          method: string | null
        }
        local_ip: string | null
        wan_ip: string | null
      }
      device: {
        serial: string
        serial_short: string
        model: string
        hostname: string
      }
      os: string
      kernel: string
      lelamp_version: string
      fan: {
        available: boolean
        rpm?: number
        pwm_percent?: number
        mode?: string
      }
    }>('/system/info'),

  // Fan control
  getFanStatus: () =>
    fetchApi<{
      success: boolean
      available: boolean
      rpm?: number
      pwm?: number
      pwm_percent?: number
      mode?: string
      mode_value?: number
      temperature?: number
    }>('/system/fan/status'),

  setFanSpeed: (percent: number) =>
    fetchApi<{
      success: boolean
      speed?: number
      mode?: string
      rpm?: number
      error?: string
    }>(`/system/fan/speed?percent=${percent}`, { method: 'POST' }),

  setFanAuto: () =>
    fetchApi<{
      success: boolean
      mode?: string
      message?: string
      error?: string
    }>('/system/fan/auto', { method: 'POST' }),
}

// Auth API
export const authApi = {
  getConfig: () =>
    fetchApi<{
      success: boolean
      enabled: boolean
      localBypass: boolean
      clerkPublishableKey: string | null
    }>('/auth/config'),

  getStatus: () =>
    fetchApi<{
      success: boolean
      authRequired: boolean
      localBypassEnabled: boolean
    }>('/auth/status'),
}

// Characters/Personality API
export const charactersApi = {
  list: () =>
    fetchApi<{
      success: boolean
      characters: Array<{
        file: string
        name: string
        description: string
      }>
    }>('/characters/'),

  getCurrent: () =>
    fetchApi<{
      success: boolean
      character: {
        name: string
        description: string
        voice_model: string
        speech_style: string
      }
    }>('/characters/current'),

  update: (characterFile: string) =>
    fetchApi<{
      success: boolean
      message: string
      character: {
        name: string
        description: string
      }
    }>('/characters/', {
      method: 'POST',
      body: JSON.stringify({ character_file: characterFile }),
    }),
}
