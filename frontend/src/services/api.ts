
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

export async function getDocuments(): Promise<DocumentInfo[]> {
  const response = await fetch("/api/documents");

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

  const response = await fetch("/api/documents/upload", {
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
  const response = await fetch("/api/questions/ask", {
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
