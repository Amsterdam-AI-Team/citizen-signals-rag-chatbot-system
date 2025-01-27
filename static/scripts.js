// 1. Variable Declarations: Manage DOM references and state variables

const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const newSessionButton = document.getElementById('new-session-button');
const modelSwitcher = document.getElementById('model-switcher');
const dropdownMenu = document.getElementById('dropdown-menu');
const dropdownButtons = dropdownMenu.querySelectorAll('button');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const topBar = document.getElementById('top-bar');
const imageUploadButton = document.getElementById('image-upload-button');
const imageUploadInput = document.getElementById('image-upload');

let sessionActive = false;
let currentModel = "ChatGPT 4o";
let currentReader = null;
let isStreaming = false;
let base64Image = null;

// 2. Event Listener Setup: Define how the script interacts with user actions

// Toggle the sidebar visibility when the toggle button is clicked
sidebarToggle.addEventListener('click', toggleSidebar);

// Adjust the send button and textarea height based on user input
userInput.addEventListener('input', handleUserInput);
userInput.addEventListener('keydown', handleUserInputKeydown);

// Handle the send button click to send the user's message
sendButton.addEventListener('click', handleSendButtonClick);

// Handle new session button click to reset the chat session
newSessionButton.addEventListener('click', handleNewSessionButtonClick);

// Toggle the model dropdown menu visibility
modelSwitcher.addEventListener('click', toggleDropdownMenu);

// Update the current model based on the dropdown selection
dropdownButtons.forEach(button => {
    button.addEventListener('click', handleDropdownSelection);
});

// Close the model dropdown if a click happens outside of it
document.addEventListener('click', closeDropdownMenuOnClick);

// Handle clicks on the document to manage bot actions and audio streaming
document.addEventListener('click', handleDocumentClick);

// Trigger the file input when the upload button is clicked
imageUploadButton.addEventListener('click', triggerImageUpload);

// Handle image file selection and preview
imageUploadInput.addEventListener('change', handleImageSelection);

// Handle session clearance when page is refreshed
window.addEventListener('beforeunload', clearSessionOnUnload);

// 3. Utility Functions: Helper functions to perform specific tasks


// Display a message in the chat interface.
function displayMessage(text, sender, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    if (sender === 'bot') {
        const modelIconDiv = document.createElement('div');
        modelIconDiv.className = 'model-icon';

        let modelIconSrc;
        switch (currentModel.toLowerCase()) {
            case 'chatgpt 4o':
                modelIconSrc = 'static/images/person_icon.png';
                break;
            default:
                modelIconSrc = ''; // Fallback if the model is not recognized
        }

        modelIconDiv.innerHTML = `<img src="${modelIconSrc}" alt="${currentModel} Icon" style="width: 100%; height: 100%; object-fit: cover;">`;
        if (isLoading) {
            modelIconDiv.classList.add('pulsing'); // Add pulsing animation for loading state
        }
        messageDiv.appendChild(modelIconDiv);

        const messageTextDiv = document.createElement('div');
        messageTextDiv.className = 'message-text';
        messageTextDiv.innerHTML = isLoading ? '' : window.marked.parse(text); // Display the message text
        messageDiv.appendChild(messageTextDiv);
    } else {
        messageDiv.innerHTML = window.marked.parse(text); // Display the user's message
    }

    messagesDiv.appendChild(messageDiv); // Add the message to the chat

    // Ensure the scroll happens after the message has been added, with smooth scrolling
    setTimeout(() => {
        messagesDiv.scrollTo({
            top: messagesDiv.scrollHeight,
            behavior: 'smooth' // Enables smooth scrolling
        });
    }, 100); // Adding a short delay to ensure DOM update completes

    // Activate new session button if there are messages
    if (messagesDiv.children.length > 0) {
        newSessionButton.classList.add('active');
        newSessionButton.disabled = false;
    }

    return messageDiv; // Return the created message element
}



// Function to add buttons to bot messages
function addBotButtons(messageDiv, text) {
    const botButtonsContainer = document.createElement('div');
    botButtonsContainer.className = 'bot-buttons-container';

    const botButtonsDiv = document.createElement('div');
    botButtonsDiv.className = 'bot-buttons';

    const readButton = document.createElement('button');
    readButton.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
    readButton.title = 'Lees voor'; // Read aloud button

    const likeButton = document.createElement('button');
    likeButton.innerHTML = '<i class="fa fa-thumbs-up"></i>';
    likeButton.title = 'Goede reactie'; // Like button

    const dislikeButton = document.createElement('button');
    dislikeButton.innerHTML = '<i class="fa fa-thumbs-down"></i>';
    dislikeButton.title = 'Slecte reactie'; // Dislike button

    const copyButton = document.createElement('button');
    copyButton.innerHTML = '<i class="fa fa-copy"></i>';
    copyButton.title = 'Kopiëren'; // Copy button

    // Handle copy button click to copy text to clipboard
    copyButton.addEventListener('click', () => {
        if (copyButton.querySelector('i').classList.contains('fa-copy')) {
            navigator.clipboard.writeText(text).then(() => {
                copyButton.innerHTML = '<i class="fa fa-check"></i>'; // Show check mark when copied
            });
        } else {
            copyButton.innerHTML = '<i class="fa fa-copy"></i>'; // Reset to copy icon
        }
    });

    botButtonsDiv.appendChild(readButton);
    botButtonsDiv.appendChild(likeButton);
    botButtonsDiv.appendChild(dislikeButton);
    botButtonsDiv.appendChild(copyButton);

    botButtonsContainer.appendChild(botButtonsDiv);
    messageDiv.appendChild(botButtonsContainer);
}

// 4. Main Functions: Core features and logic

// Function to handle toggling the sidebar
function toggleSidebar() {
    sidebar.classList.toggle('visible');
    if (sidebar.classList.contains('visible')) {
        document.body.style.marginLeft = '250px'; // Shift the body to the right
        topBar.classList.remove('shifted'); // Adjust the top bar
        sidebarToggle.classList.add('active'); // Indicate the sidebar is active
    } else {
        document.body.style.marginLeft = '0'; // Reset the body margin
        topBar.classList.add('shifted'); // Reapply the shifted class
        sidebarToggle.classList.remove('active'); // Indicate the sidebar is inactive
    }
}

// Function to handle user input
function handleUserInput() {
    // Enable send button if text or an image is present
    sendButton.disabled = !(userInput.value.trim() || base64Image);
    sendButton.classList.toggle('active', !sendButton.disabled); // Style the button if it's active
    userInput.style.height = 'auto'; // Reset height to adjust dynamically
    userInput.style.height = `${userInput.scrollHeight}px`; // Adjust height to fit content
}

// Function to handle keydown events in the user input
function handleUserInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent newline from being added
        sendButton.click(); // Simulate a click on the send button
    }
}

// Function to handle sending the user's message
async function handleSendButtonClick() {
    const userMessage = userInput.value.trim(); // Get the user's message
    if (userMessage || base64Image) { // Check if there is a message or an image
        const imageContainerDiv = document.getElementById('image-container');
        imageContainerDiv.innerHTML = ''; // Clear the image container (removes the preview image)
        document.getElementById('image-preview').style.display = 'none'; // Hide the image preview section
        imageUploadInput.value = ''; // Reset the file input so it can accept new images

        // Hide placeholder message and SVG container
        document.getElementById('placeholder-message').style.display = 'none';
        document.getElementById('svg-container').style.display = 'none';

        // Display image if exists
        if (base64Image) {
            displayImageMessage(base64Image, 'user'); // Display image first
        }

        // Now display the user's text message (if there is one)
        if (userMessage) {
            displayMessage(userMessage, 'user'); // Display text message
        }

        userInput.value = ''; // Clear the text input
        sendButton.disabled = true; // Disable the send button
        userInput.style.height = 'auto'; // Reset input height

        const loadingMessageDiv = displayMessage("", 'bot', true); // Show loading message

        // Send the message and image to the server
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                model: currentModel,
                session: sessionActive,
                image: base64Image // Retain base64Image for sending to the server
            }),
        });

        // Clear the image data after the fetch is completed
        base64Image = null; // Clear the image data only after the fetch

        const result = await response.json(); // Parse the server response

        // Assume result.response is now an array of strings
        const responses = Array.isArray(result.response) ? result.response : [result.response];

        // Remove the initial loading message since we'll add multiple messages
        if (loadingMessageDiv) {
            loadingMessageDiv.remove();
        }

        // Process each response message with a delay
        for (const responseText of responses) {
            const messageDiv = displayMessage("", 'bot', true); // Create a new message div for each response
            const messageTextDiv = messageDiv.querySelector('.message-text');
            if (messageTextDiv) {
                messageTextDiv.innerHTML = window.marked.parse(responseText); // Parse and display each response
            }

            const modelIconDiv = messageDiv.querySelector('.model-icon');
            if (modelIconDiv) {
                modelIconDiv.classList.remove('pulsing'); // Stop pulsing animation
            }

            addBotButtons(messageDiv, responseText); // Add interaction buttons to each message

            // Delay of 2 seconds between messages
            await new Promise(resolve => setTimeout(resolve, 3000));
        }

        // Enable the new session button if there are messages
        if (messagesDiv.children.length > 0) {
            newSessionButton.classList.add('active');
            newSessionButton.disabled = false;
        }
    }
}

// Function to display images
function displayImageMessage(base64Image, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender} image-message`; // Add an extra 'image-message' class

    const imgElement = document.createElement('img');
    imgElement.src = `data:image/jpeg;base64,${base64Image}`;
    imgElement.style.maxWidth = '200px'; // Limit the size of the image
    imgElement.style.borderRadius = '10px'; // Add some border radius for a nice look
    imgElement.style.display = 'block';

    messageDiv.appendChild(imgElement);
    messagesDiv.appendChild(messageDiv); // Add the image message to the chat
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the latest message

    return messageDiv; // Return the created image message element
}

// Function to handle starting a new session
async function handleNewSessionButtonClick() {
    if (messagesDiv.children.length > 0) {
        await fetch('/new_session', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        messagesDiv.innerHTML = ''; // Clear the messages
        sessionActive = false; // Reset session state
        newSessionButton.classList.remove('active'); // Deactivate the button
        newSessionButton.disabled = true; // Disable the button
    }
}

// Function to delete session file when page is refreshed
function clearSessionOnUnload() {
    navigator.sendBeacon('/clear-session');
}

// Function to toggle the model dropdown menu
function toggleDropdownMenu() {
    dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
}

// Function to handle dropdown menu selection
function handleDropdownSelection(event) {
    const button = event.target.closest('button');
    if (button && dropdownMenu.contains(button)) {
        const model = button.getAttribute('data-model'); // Update the current model
        currentModel = model;
        modelSwitcher.innerHTML = `${currentModel} <i class="fas fa-chevron-down"></i>`; // Update the button text
        dropdownButtons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected'); // Select the clicked button
        dropdownMenu.style.display = 'none'; // Hide the dropdown menu
    }
}

// Function to close dropdown menu if click happens outside of it
function closeDropdownMenuOnClick(e) {
    if (!modelSwitcher.contains(e.target)) {
        dropdownMenu.style.display = 'none'; // Hide the dropdown menu
    }
}

// Function to trigger image upload
function triggerImageUpload() {
    imageUploadInput.click();
}

// Function to handle image selection and preview
function handleImageSelection() {
    const file = imageUploadInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
            base64Image = reader.result.split(',')[1]; // Convert image to base64

            const imgElement = document.createElement('img');
            imgElement.src = `data:image/jpeg;base64,${base64Image}`; // Display base64 image

            const imageContainerDiv = document.getElementById('image-container');
            imageContainerDiv.innerHTML = ''; // Clear previous content
            imageContainerDiv.appendChild(imgElement); // Add new image

            const deleteIcon = document.createElement('button');
            deleteIcon.className = 'icon-button delete-icon';
            deleteIcon.innerHTML = '<i class="fas fa-times"></i>'; // Add delete icon

            imageContainerDiv.appendChild(deleteIcon);
            document.getElementById('image-preview').style.display = 'block'; // Show the preview

            // Enable send button if image is present
            sendButton.disabled = !(userInput.value.trim() || base64Image);
            sendButton.classList.toggle('active', !sendButton.disabled);

            // Allow the user to remove the uploaded image
            deleteIcon.addEventListener('click', () => {
                imageContainerDiv.innerHTML = ''; // Clear the image container
                document.getElementById('image-preview').style.display = 'none'; // Hide the preview
                base64Image = null; // Clear the base64 image data
                imageUploadInput.value = ''; // Reset the file input

                // Disable send button if no text or image is present
                sendButton.disabled = !(userInput.value.trim() || base64Image);
                sendButton.classList.toggle('active', !sendButton.disabled);
            });
        };
        reader.readAsDataURL(file); // Read the file as a data URL
    }
}

