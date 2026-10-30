
export interface DocumentInfo {
  document_id: string;
  size_bytes: number;
}

export interface UploadResponse {
  filename: string;
  document_id: string;
  message: string;
  extracted_characters: number;
}

export interface QuestionRequest {
  question: string;
  top_k?: number;
}

export interface SourceInfo {
  filename: string;
  document_id: string;
  chunk_index: number | null;
}

export interface AnswerResponse {
  answer: string;
  sources: SourceInfo[];
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  username: string;
}

const ACCESS_TOKEN_KEY = 'access_token'

export function getAccessToken(): string | null {
  return sessionStorage.getItem(ACCESS_TOKEN_KEY)
}

export function setAccessToken(token: string): void {
  sessionStorage.setItem(ACCESS_TOKEN_KEY, token)
}

export function clearAccessToken(): void {
  sessionStorage.removeItem(ACCESS_TOKEN_KEY)
}

function authorizedHeaders(
  headers: HeadersInit = {},
): Headers {
  const result = new Headers(headers)
  const token = getAccessToken()
  if (token) {
    result.set('Authorization', `Bearer ${token}`)
  }
  return result
}

async function authorizedFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const response = await fetch(input, {
    ...init,
    headers: authorizedHeaders(init.headers),
  })

  if (response.status === 401) {
    clearAccessToken()
    window.dispatchEvent(new Event('auth-expired'))
  }

  return response
}

export async function register(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const response = await fetch('/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })

  if (!response.ok) {
    throw new Error('Failed to create account.')
  }

  return (await response.json()) as AuthResponse
}

export async function login(
  username: string,
  password: string,
): Promise<AuthResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })

  if (!response.ok) {
    throw new Error('Invalid username or password.')
  }

  return (await response.json()) as AuthResponse
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const response = await authorizedFetch('/api/documents')

  if (!response.ok) {
    throw new Error("Failed to fetch documents.");
  }

  return (await response.json()) as DocumentInfo[];
}

export async function uploadDocument(
  file: File,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await authorizedFetch("/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to upload document.");
  }

  return (await response.json()) as UploadResponse;
}

export async function askQuestion(
  request: QuestionRequest,
): Promise<AnswerResponse> {
  const response = await authorizedFetch("/api/questions/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error("Failed to get an answer.");
  }

  return (await response.json()) as AnswerResponse;
}
