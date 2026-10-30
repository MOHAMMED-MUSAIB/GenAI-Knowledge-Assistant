
import { useEffect, useState, type FormEvent } from 'react'
import {
  askQuestion,
  clearAccessToken,
  getDocuments,
  getAccessToken,
  login,
  register,
  setAccessToken,
  uploadDocument,
  type AnswerResponse,
  type DocumentInfo,
} from './services/api'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => getAccessToken() !== null,
  )
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState('')
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadMessage, setUploadMessage] = useState('')
  const [isUploading, setIsUploading] = useState(false)

  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AnswerResponse | null>(null)
  const [isAsking, setIsAsking] = useState(false)
  const [questionError, setQuestionError] = useState('')

  useEffect(() => {
    function handleAuthExpired() {
      setIsAuthenticated(false)
      setDocuments([])
      setAnswer(null)
    }

    window.addEventListener('auth-expired', handleAuthExpired)
    return () => window.removeEventListener('auth-expired', handleAuthExpired)
  }, [])

  useEffect(() => {
    if (!isAuthenticated) {
      return
    }

    async function loadDocuments() {
      try {
        const result = await getDocuments()
        setDocuments(result)
        setError('')
      } catch {
        setError(
          'Could not load documents. Check that the backend is running.',
        )
      }
    }

    void loadDocuments()
  }, [isAuthenticated])

  async function handleAuthentication(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setIsAuthenticating(true)
    setAuthError('')

    try {
      const response =
        authMode === 'register'
          ? await register(username, password)
          : await login(username, password)
      setAccessToken(response.access_token)
      setIsAuthenticated(true)
      setPassword('')
    } catch (requestError) {
      setAuthError(
        requestError instanceof Error
          ? requestError.message
          : 'Authentication failed.',
      )
    } finally {
      setIsAuthenticating(false)
    }
  }

  function handleLogout() {
    clearAccessToken()
    setIsAuthenticated(false)
    setDocuments([])
    setAnswer(null)
  }

  if (!isAuthenticated) {
    return (
      <main>
        <h1>Document Library</h1>
        <section>
          <h2>{authMode === 'register' ? 'Create account' : 'Sign in'}</h2>
          <form onSubmit={handleAuthentication}>
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              minLength={3}
              maxLength={64}
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
              disabled={isAuthenticating}
            />
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              minLength={12}
              maxLength={128}
              autoComplete={
                authMode === 'register' ? 'new-password' : 'current-password'
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              disabled={isAuthenticating}
            />
            <button type="submit" disabled={isAuthenticating}>
              {isAuthenticating
                ? 'Please wait...'
                : authMode === 'register'
                  ? 'Create account'
                  : 'Sign in'}
            </button>
          </form>
          {authError && <p role="alert">{authError}</p>}
          <button
            type="button"
            onClick={() => {
              setAuthError('')
              setAuthMode(authMode === 'register' ? 'login' : 'register')
            }}
          >
            {authMode === 'register'
              ? 'Already have an account? Sign in'
              : 'Need an account? Register'}
          </button>
        </section>
      </main>
    )
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!selectedFile) {
      setUploadMessage('Choose a file to upload.')
      return
    }

    const form = event.currentTarget

    setIsUploading(true)
    setUploadMessage('')
    setError('')

    try {
      const uploaded = await uploadDocument(selectedFile)

      setUploadMessage(`${uploaded.filename} uploaded successfully.`)
      setSelectedFile(null)
      form.reset()

      try {
        const updatedDocuments = await getDocuments()
        setDocuments(updatedDocuments)
      } catch {
        setError(
          'The document uploaded, but the document list could not be refreshed.',
        )
      }
    } catch (uploadError) {
      setUploadMessage(
        uploadError instanceof Error
          ? uploadError.message
          : 'Could not upload document.',
      )
    } finally {
      setIsUploading(false)
    }
  }

  async function handleAskQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const trimmedQuestion = question.trim()

    if (!trimmedQuestion) {
      setQuestionError('Please enter a question.')
      return
    }

    setIsAsking(true)
    setQuestionError('')
    setAnswer(null)

    try {
      const result = await askQuestion({
        question: trimmedQuestion,
        top_k: 3,
      })

      setAnswer(result)
    } catch (questionRequestError) {
      setQuestionError(
        questionRequestError instanceof Error
          ? questionRequestError.message
          : 'Could not get an answer. Please try again.',
      )
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main>
      <h1>Document Library</h1>
      <button type="button" onClick={handleLogout}>
        Sign out
      </button>

      {error && <p role="alert">{error}</p>}

      <section>
        <h2>Upload a document</h2>

        <form onSubmit={handleUpload}>
          <label htmlFor="document-file">
            Choose a PDF, DOCX, or TXT file (maximum 10 MB)
          </label>

          <input
            id="document-file"
            type="file"
            accept=".pdf,.docx,.txt"
            disabled={isUploading}
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] ?? null)
              setUploadMessage('')
            }}
          />

          <button
            type="submit"
            disabled={!selectedFile || isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload document'}
          </button>
        </form>

        {uploadMessage && <p role="status">{uploadMessage}</p>}
      </section>

      <section>
        <h2>Uploaded documents</h2>

        {documents.length === 0 && !error && (
          <p>No documents found yet.</p>
        )}

        {documents.length > 0 && (
          <ul>
            {documents.map((document) => (
              <li key={document.document_id}>
                {document.document_id} - {document.size_bytes} bytes
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2>Ask a Question</h2>

        <form onSubmit={handleAskQuestion}>
          <label htmlFor="question-input">
            Ask something about your uploaded documents
          </label>

          <input
            id="question-input"
            type="text"
            value={question}
            onChange={(event) => {
              setQuestion(event.target.value)
              setQuestionError('')
            }}
            placeholder="e.g. What skills are listed in my resume?"
            disabled={isAsking}
          />

          <button
            type="submit"
            disabled={isAsking || !question.trim()}
          >
            {isAsking ? 'Thinking...' : 'Ask'}
          </button>
        </form>

        {questionError && (
          <p role="alert">{questionError}</p>
        )}

        {answer && (
          <div>
            <h3>Answer</h3>
            <p>{answer.answer}</p>

            <h3>Sources</h3>

            {answer.sources.length > 0 ? (
              <ul>
                {answer.sources.map((source, index) => (
                  <li
                    key={`${source.document_id}-${source.chunk_index}-${index}`}
                  >
                    <strong>{source.filename}</strong>
                    {' — '}
                    Chunk {source.chunk_index ?? 'N/A'}
                    {' — '}
                    Document ID: {source.document_id}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No sources were returned.</p>
            )}
          </div>
        )}
      </section>
    </main>
  )
}

export default App
