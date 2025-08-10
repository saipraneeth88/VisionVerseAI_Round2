document.addEventListener('DOMContentLoaded', () => {

    /* -------------------------------
       DOM ELEMENT REFERENCES
    ------------------------------- */
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('video-input');
    const fileNameDisplay = document.querySelector('.file-name');
    const videoPreview = document.getElementById('video-preview');
    const summaryContent = document.getElementById('summary-content');
    const chatHistoryDiv = document.getElementById('chat-history');
    const chatForm = document.getElementById('chat-form');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');

    /* -------------------------------
       APPLICATION STATE VARIABLES
    ------------------------------- */
    let uploadedVideoFile = null;          // Stores the selected video file
    let conversationHistory = [];          // Stores chat history (role/content)

    /* -------------------------------
       EVENT LISTENERS
    ------------------------------- */
    videoInput.addEventListener('change', handleFileSelect);
    uploadForm.addEventListener('submit', handleVideoAnalysis);
    chatForm.addEventListener('submit', handleChatMessage);

    /* -------------------------------
       HANDLE VIDEO FILE SELECTION
    ------------------------------- */
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        uploadedVideoFile = file;
        fileNameDisplay.textContent = file.name;

        // Preview the selected video
        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        videoPreview.style.display = 'block';

        // Reset UI & chat history for new video
        conversationHistory = [];
        chatHistoryDiv.innerHTML = '';
        summaryContent.innerHTML = `<p>Click "Analyze Video" to begin.</p>`;
    }

    /* -------------------------------
       HANDLE VIDEO ANALYSIS REQUEST
       (First Interaction)
    ------------------------------- */
    async function handleVideoAnalysis(event) {
        event.preventDefault();
        const initialPrompt = "Provide a detailed, structured summary of this video and highlight any potential safety violations.";

        // Reset history for a new analysis
        conversationHistory = [];
        await callInferAPI(initialPrompt, true);
    }

    /* -------------------------------
       HANDLE CHAT FOLLOW-UP QUESTIONS
    ------------------------------- */
    async function handleChatMessage(event) {
        event.preventDefault();
        const question = questionInput.value.trim();
        if (!question) return;

        appendMessage(question, 'user-message');
        questionInput.value = '';

        await callInferAPI(question, false);
    }

    /* -------------------------------
       API CALL TO BACKEND /infer
       Handles both first analysis & follow-up Q&A
    ------------------------------- */
    async function callInferAPI(prompt, isFirstAnalysis) {
        if (!uploadedVideoFile) {
            alert("Please select a video file first.");
            return;
        }

        // Show loading animation
        const loadingHTML = `<div class="loading-spinner"></div><p><i>Analyzing, please wait...</i></p>`;
        if (isFirstAnalysis) {
            summaryContent.innerHTML = loadingHTML;
        } else {
            appendMessage('<div class="loading-spinner"></div>', 'ai-message thinking');
        }

        // Disable chat input while processing
        questionInput.disabled = true;
        sendButton.disabled = true;

        // Prepare request payload
        const formData = new FormData();
        formData.append('video', uploadedVideoFile);
        formData.append('prompt', prompt);
        formData.append('history', JSON.stringify(conversationHistory)); // Maintain context

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                body: formData
            });

            const resultText = await response.text();
            if (!response.ok) throw new Error(resultText);

            // Save conversation in memory
            conversationHistory.push({ role: "user", content: prompt });
            conversationHistory.push({ role: "assistant", content: resultText });

            // Format AI output as HTML
            const formattedHtml = marked.parse(resultText);

            if (isFirstAnalysis) {
                summaryContent.innerHTML = formattedHtml;
            } else {
                const thinkingBubble = document.querySelector('.thinking');
                if (thinkingBubble) thinkingBubble.remove();
                appendMessage(formattedHtml, 'ai-message');
            }

        } catch (error) {
            const errorMessage = `<strong>Error:</strong> ${error.message || "Failed to process request."}`;
            if (isFirstAnalysis) {
                summaryContent.innerHTML = errorMessage;
            } else {
                const thinkingBubble = document.querySelector('.thinking');
                if (thinkingBubble) thinkingBubble.remove();
                appendMessage(errorMessage, 'ai-message');
            }
        } finally {
            // Re-enable input
            questionInput.disabled = false;
            sendButton.disabled = false;
            questionInput.focus();
        }
    }

    /* -------------------------------
       APPEND MESSAGE TO CHAT UI
    ------------------------------- */
    function appendMessage(content, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = className;

        if (className === 'user-message') {
            messageDiv.textContent = content;
        } else {
            messageDiv.innerHTML = content;
        }

        chatHistoryDiv.appendChild(messageDiv);
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }
});
