
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import * as api from './services/api'

vi.mock('./services/api', () => ({
  clearAccessToken: vi.fn(),
  getDocuments: vi.fn().mockResolvedValue([]),
  getAccessToken: vi.fn(),
  login: vi.fn(),
  register: vi.fn(),
  setAccessToken: vi.fn(),
  uploadDocument: vi.fn(),
  askQuestion: vi.fn(),
}))

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getAccessToken).mockReturnValue('test-token')
    vi.mocked(api.getDocuments).mockResolvedValue([])
  })

  it('signs in before showing the document library', async () => {
    vi.mocked(api.getAccessToken).mockReturnValue(null)
    vi.mocked(api.login).mockResolvedValue({
      access_token: 'signed-in-token',
      token_type: 'bearer',
      username: 'test-user',
    })

    render(<App />)
    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'test-user' },
    })
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'a-secure-password-123' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^sign in$/i }))

    expect(await screen.findByText(/upload a document/i)).toBeInTheDocument()
    expect(api.login).toHaveBeenCalledWith(
      'test-user',
      'a-secure-password-123',
    )
    expect(api.setAccessToken).toHaveBeenCalledWith('signed-in-token')
  })

  it('creates an account and stores the access token', async () => {
    vi.mocked(api.getAccessToken).mockReturnValue(null)
    vi.mocked(api.register).mockResolvedValue({
      access_token: 'new-account-token',
      token_type: 'bearer',
      username: 'new-user',
    })

    render(<App />)
    fireEvent.click(
      screen.getByRole('button', { name: /need an account\? register/i }),
    )
    fireEvent.change(screen.getByLabelText(/username/i), {
      target: { value: 'new-user' },
    })
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'a-secure-password-123' },
    })
    fireEvent.click(screen.getByRole('button', { name: /^create account$/i }))

    expect(await screen.findByText(/upload a document/i)).toBeInTheDocument()
    expect(api.register).toHaveBeenCalledWith(
      'new-user',
      'a-secure-password-123',
    )
    expect(api.setAccessToken).toHaveBeenCalledWith('new-account-token')
  })

  it('renders the document upload section', async () => {
    render(<App />)

    expect(
      screen.getByText(/upload a document/i),
    ).toBeInTheDocument()

    await screen.findByText(/no documents found yet/i)
  })

  it('disables upload until a file is selected', async () => {
    render(<App />)

    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled()
    })

    expect(
      screen.getByRole('button', { name: /upload document/i }),
    ).toBeDisabled()
  })

  it('uploads a selected file successfully', async () => {
    vi.mocked(api.uploadDocument).mockResolvedValue({
      filename: 'sample.txt',
      document_id: 'sample-id',
      message: 'Document uploaded successfully.',
      extracted_characters: 23,
    })

    render(<App />)

    const file = new File(['Sample document content'], 'sample.txt', {
      type: 'text/plain',
    })

    fireEvent.change(screen.getByLabelText(/choose a pdf/i), {
      target: { files: [file] },
    })

    fireEvent.click(screen.getByRole('button', { name: /upload document/i }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      /sample\.txt uploaded successfully/i,
    )

    expect(api.uploadDocument).toHaveBeenCalledWith(file)
  })

  it('displays an error when upload fails', async () => {
    vi.mocked(api.uploadDocument).mockRejectedValue(
      new Error('Unsupported file type.'),
    )

    render(<App />)

    const file = new File(['image content'], 'photo.png', {
      type: 'image/png',
    })

    fireEvent.change(screen.getByLabelText(/choose a pdf/i), {
      target: { files: [file] },
    })

    fireEvent.click(screen.getByRole('button', { name: /upload document/i }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      'Unsupported file type.',
    )

    expect(
      screen.getByRole('button', { name: /upload document/i }),
    ).toBeEnabled()
  })

  it('displays an answer and its source after asking a question', async () => {
    vi.mocked(api.askQuestion).mockResolvedValue({
      answer: 'Qdrant stores vectors for semantic search.',
      sources: [
        {
          filename: 'sample.pdf',
          document_id: 'sample-id',
          chunk_index: 0,
        },
      ],
    })

    render(<App />)

    const questionInput = screen.getByLabelText(
      /ask something about your uploaded documents/i,
    )

    fireEvent.change(questionInput, {
      target: { value: 'What is Qdrant?' },
    })

    fireEvent.click(screen.getByRole('button', { name: /^ask$/i }))

    expect(
      await screen.findByText('Qdrant stores vectors for semantic search.'),
    ).toBeInTheDocument()

    expect(screen.getByText('sample.pdf')).toBeInTheDocument()
    expect(screen.getByText(/Document ID: sample-id/)).toBeInTheDocument()

    expect(api.askQuestion).toHaveBeenCalledWith({
      question: 'What is Qdrant?',
      top_k: 3,
    })
  })

  it('displays an error when asking a question fails', async () => {
    vi.mocked(api.askQuestion).mockRejectedValue(
      new Error('Could not get an answer. Please try again.'),
    )

    render(<App />)

    fireEvent.change(
      screen.getByLabelText(/ask something about your uploaded documents/i),
      { target: { value: 'What is Qdrant?' } },
    )

    fireEvent.click(screen.getByRole('button', { name: /^ask$/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not get an answer. Please try again.',
    )

    expect(
      screen.getByRole('button', { name: /^ask$/i }),
    ).toBeEnabled()
  })
})
