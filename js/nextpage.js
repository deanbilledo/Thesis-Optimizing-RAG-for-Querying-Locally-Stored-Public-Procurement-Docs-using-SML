/**
 * WESDEABEN PDF Summarizer JavaScript
 * Adds interactivity to the PDF summary page with model selection
 */

document.addEventListener('DOMContentLoaded', function() {
    // Model selection elements
    const modelSelect = document.getElementById('model-select');
    const modelStatus = document.getElementById('model-status');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const useOllamaCheckbox = document.getElementById('use-ollama');
    const responseContainer = document.getElementById('response-container');
    const responseText = document.getElementById('response-text');
    const modelUsed = document.getElementById('model-used');
    const sourcesContainer = document.getElementById('sources-container');
    const sourcesList = document.getElementById('sources-list');
    const loading = document.getElementById('loading');
    
    let availableModels = [];
    let defaultModel = null;
    
    // Load available models on page load
    loadAvailableModels();
    
    // Set up event listeners
    if (sendButton) {
        sendButton.addEventListener('click', handleQuery);
    }
    if (queryInput) {
        queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleQuery();
            }
        });
    }
    if (modelSelect) {
        modelSelect.addEventListener('change', updateModelStatus);
    }
    
    // Load available models from the server
    async function loadAvailableModels() {
        try {
            const response = await fetch('/models');
            const data = await response.json();
            
            availableModels = data.models || [];
            defaultModel = data.default_model;
            
            populateModelDropdown();
            updateModelStatus();
            
        } catch (error) {
            console.error('Error loading models:', error);
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Error loading models</option>';
            }
            if (modelStatus) {
                modelStatus.textContent = 'Failed to load models';
                modelStatus.className = 'model-status error';
            }
        }
    }
    
    // Populate the model dropdown
    function populateModelDropdown() {
        if (!modelSelect) return;
        
        modelSelect.innerHTML = '';
        
        if (availableModels.length === 0) {
            modelSelect.innerHTML = '<option value="">No models available</option>';
            return;
        }
        
        // Add default option
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        
        // Add available models
        availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} ${model.available ? '✓' : '✗'}`;
            option.disabled = !model.available;
            
            // Select default model if available
            if (model.id === defaultModel && model.available) {
                option.selected = true;
            }
            
            modelSelect.appendChild(option);
        });
        
        // Update status after populating
        updateModelStatus();
    }
    
    // Update model status display
    function updateModelStatus() {
        if (!modelStatus || !modelSelect) return;
        
        const selectedModelId = modelSelect.value;
        
        if (!selectedModelId) {
            modelStatus.textContent = 'Please select a model';
            modelStatus.className = 'model-status warning';
            return;
        }
        
        const selectedModel = availableModels.find(m => m.id === selectedModelId);
        
        if (selectedModel) {
            if (selectedModel.available) {
                modelStatus.textContent = `✓ ${selectedModel.description}`;
                modelStatus.className = 'model-status available';
            } else {
                modelStatus.textContent = `✗ ${selectedModel.description} (Not available)`;
                modelStatus.className = 'model-status unavailable';
            }
        }
    }
    
    // Handle query submission
    async function handleQuery() {
        if (!queryInput || !sendButton) return;
        
        const query = queryInput.value.trim();
        const selectedModel = modelSelect ? modelSelect.value : '';
        const useOllama = useOllamaCheckbox ? useOllamaCheckbox.checked : false;
        
        if (!query) {
            alert('Please enter a question');
            return;
        }
        
        if (!useOllama && !selectedModel) {
            alert('Please select a model or use Ollama fallback');
            return;
        }
        
        // Show loading state
        if (loading) loading.style.display = 'flex';
        if (responseContainer) responseContainer.style.display = 'none';
        sendButton.disabled = true;
        
        try {
            const requestBody = {
                query: query,
                use_ollama: useOllama
            };
            
            // Add model_id if not using Ollama
            if (!useOllama && selectedModel) {
                requestBody.model_id = selectedModel;
            }
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });
            
            const data = await response.json();
            
            // Hide loading
            if (loading) loading.style.display = 'none';
            
            if (data.error) {
                showError(data.error);
            } else {
                showResponse(data);
            }
            
        } catch (error) {
            console.error('Error querying model:', error);
            if (loading) loading.style.display = 'none';
            showError('Failed to connect to the server. Please check if the Flask app is running.');
        } finally {
            sendButton.disabled = false;
        }
    }
    
    // Show successful response
    function showResponse(data) {
        if (!responseContainer || !responseText) return;
        
        responseContainer.style.display = 'block';
        responseText.textContent = data.response || 'No response received';
        
        // Show model used
        if (modelUsed) {
            if (data.model_used) {
                modelUsed.textContent = data.model_used;
                modelUsed.style.display = 'inline-block';
            } else {
                modelUsed.style.display = 'none';
            }
        }
        
        // Show sources if available
        if (sourcesContainer && sourcesList) {
            if (data.sources && data.sources.length > 0) {
                sourcesContainer.style.display = 'block';
                sourcesList.innerHTML = '';
                data.sources.forEach(source => {
                    const li = document.createElement('li');
                    li.textContent = source;
                    sourcesList.appendChild(li);
                });
            } else {
                sourcesContainer.style.display = 'none';
            }
        }
        
        // Scroll to response
        responseContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Show error message
    function showError(errorMessage) {
        if (!responseContainer || !responseText) return;
        
        responseContainer.style.display = 'block';
        responseText.innerHTML = `<div class="error-message">Error: ${errorMessage}</div>`;
        
        if (modelUsed) modelUsed.style.display = 'none';
        if (sourcesContainer) sourcesContainer.style.display = 'none';
        
        // Scroll to response
        responseContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Handle Enter key in chat input
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });
    
    // Initialize page
    loadAvailableModels();
    
    // PDF navigation controls
    const prevButton = document.querySelector('.pdf-navigation .nav-button:first-child');
    const nextButton = document.querySelector('.pdf-navigation .nav-button:last-child');
    const pageNumberInput = document.querySelector('.page-number');
    const totalPages = 33; // From the screenshot
    
    let currentPage = 1;
    
    if (prevButton) {
        prevButton.addEventListener('click', function() {
            if (currentPage > 1) {
                currentPage--;
                updatePageDisplay();
            }
        });
    }
    
    if (nextButton) {
        nextButton.addEventListener('click', function() {
            if (currentPage < totalPages) {
                currentPage++;
                updatePageDisplay();
            }
        });
    }
    
    if (pageNumberInput) {
        pageNumberInput.addEventListener('change', function() {
            const newPage = parseInt(this.value);
            if (!isNaN(newPage) && newPage >= 1 && newPage <= totalPages) {
                currentPage = newPage;
            }
            updatePageDisplay();
        });
    }
    
    function updatePageDisplay() {
        pageNumberInput.value = currentPage;
        
        // In a real application, this would load the corresponding page
        console.log('Current page:', currentPage);
    }
    
    // PDF zoom controls
    const zoomOutButton = document.querySelector('.pdf-tools .tool-button:nth-child(1)');
    const zoomInButton = document.querySelector('.pdf-tools .tool-button:nth-child(2)');
    const fullscreenButton = document.querySelector('.pdf-tools .tool-button:nth-child(3)');
    
    let zoomLevel = 100; // percentage
    
    zoomOutButton.addEventListener('click', function() {
        if (zoomLevel > 50) {
            zoomLevel -= 10;
            updateZoom();
        }
    });
    
    zoomInButton.addEventListener('click', function() {
        if (zoomLevel < 200) {
            zoomLevel += 10;
            updateZoom();
        }
    });
    
    function updateZoom() {
        // In a real application, this would adjust the PDF zoom
        console.log('Zoom level:', zoomLevel + '%');
    }
    
    fullscreenButton.addEventListener('click', function() {
        // In a real application, this would toggle fullscreen mode
        console.log('Toggle fullscreen');
    });
    
    // Download button
    const downloadButton = document.querySelector('.download-button');
    
    downloadButton.addEventListener('click', function() {
        // In a real application, this would trigger the PDF download
        console.log('Downloading PDF');
    });
    
    // Feedback buttons
    const feedbackButtons = document.querySelectorAll('.feedback-button');
    
    feedbackButtons.forEach(button => {
        button.addEventListener('click', function() {
            // In a real application, this would send feedback
            const isPositive = this.querySelector('i').classList.contains('fa-smile');
            console.log('Feedback:', isPositive ? 'positive' : 'negative');
            
            // Visual feedback
            this.style.color = '#c1a06d';
            setTimeout(() => {
                this.style.color = '';
            }, 1000);
        });
    });
    
    // Copy button
    const copyButton = document.querySelector('.copy-button');
    
    copyButton.addEventListener('click', function() {
        // In a real application, this would copy the summary to clipboard
        console.log('Copying summary to clipboard');
        
        // Visual feedback
        this.style.color = '#c1a06d';
        setTimeout(() => {
            this.style.color = '';
        }, 1000);
    });
});
