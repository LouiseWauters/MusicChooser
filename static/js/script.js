function play_music(audioElementId, startTime) {
    const music = document.getElementById(audioElementId)
    music.currentTime = startTime;
    music.play();
}

function set_loading() {
    // Put a load spinner in the main section of the document
    document.getElementById('mainSection').innerHTML = '<div id="wrapper">\n' +
        '            <div class="simple-load-spinner"></div>\n' +
        '        </div>';
}

function fetch_content(url, elementId) {
    fetch(url, {
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
        .then(response => response.text())
        .then(data => {
            // Update the content section with the received data
            document.getElementById(elementId).innerHTML = data;
            setNextButtonDisabled(false);
            set_handlers(url)
        })
        .catch(error => {
            console.error(error);
        });
}

function set_handlers(url) {
    
    if (url === '/experiment') {
        startExperiment();
    }
    if (url === '/thanks') {
        document.getElementById("message").innerText = ""
    }
    if (url === pageOrder[pageOrder.length - 1]) {
        setElementHidden("nextButton", true);
    }
}

// Access webcam
async function init() {
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: true
    });
    handleSuccess(stream);
}

// Success
function handleSuccess(stream) {
    window.stream = stream;
    video.srcObject = stream;
}

function sendImages(userId, intervalTimeout, durationSeconds) {
    const interval = startImageSendingInterval(userId, intervalTimeout);
    setTimeout(() => stopInterval(interval), durationSeconds * 1000);
    return interval;
}

function startImageSendingInterval(userId, timeout = 1000) {
    // Continuously send image to server
    return setInterval(function() {
        context.drawImage(video, 0, 0, 640, 480);
        const data = canvas.toDataURL("image/jpeg");
        sendImage(data, userId);
    }, timeout);
}

function stopInterval(interval) {
    clearInterval(interval);
}

function sendImage(data, userId) {
    // Send a POST request to the server
    fetch('/image', {
        method: 'POST',
        headers: new Headers({'content-type': 'application/json'}),
        body: JSON.stringify({
            image: data,
            user_id: userId
        })
    })
        .then(response => response.text())
        .catch(error => {
            // Handle any errors that occur during the request
            console.error(error);
        });
}

function setNextButtonDisabled(value) {
    document.getElementById("nextButton").disabled = value;
}

function setElementHidden(elementId, value) {
    document.getElementById(elementId).hidden = value;
}

function startExperiment() {
    // Expose the video tag
    setElementHidden("videoSection", false);
    // Expose the stop button
    setElementHidden("stopButton", false);
    // Disable the next button
    setNextButtonDisabled(true);
    // Give disclaimer
    document.getElementById("message").innerText = "The experiment has begun. Please wait a few seconds for the music to start."
    // Get a user id from server
    getUserId()
        .then(getAction)
        .catch(error => {
            console.error(error);
        });
}

function endExperiment() {
    // Hide the video tag
    setElementHidden("videoSection", true);
    // Enable the next button
    setNextButtonDisabled(false);
    // Remove any text
    document.getElementById("message").innerText = ""
}

function getAction(userId) {
    // get new action
    fetch('/action?' + new URLSearchParams({
        user_id: userId
    }), {
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
        .then(response => response.text())
        .then(data => {
            // Perform the action
            if (data === 'start') {
                startProcedure(userId);
                getAction(userId);
            } else if (data === 'end') {
                endExperiment()
            } else {
                // console.log("received song:", data);
                document.getElementById("message").innerText = ""
                playNewSong(userId, data);
                sendImages(userId, 1000/IMAGE_FPS, SONG_DURATION);
            }
        })
        .catch(error => {
            console.error(error);
        });
}

function requestStop(userId) {
    // send request to stop the experiment
    fetch('/stop?' + new URLSearchParams({
        user_id: userId
    }), {
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
        .then(response => response.text())
        .then(data => {
            console.log("Requested to stop experiment", data);
        })
        .catch(error => {
            console.error(error);
        });
}

function playNewSong(userId, songFileName) {
    document.getElementById("audioSource").src = songFileName;
    document.getElementById("music").load();
    play_music("music", 0);
    setTimeout(() => {
        document.getElementById("music").pause();
        getAction(userId);
    }, SONG_DURATION * 1000);
}

function startProcedure(userId) {
    // Send enough images to fill buffer
    return sendImages(userId,1000/IMAGE_FPS, Math.max(BUFFER_SIZE/IMAGE_FPS, 20));
}

async function getUserId() {
    let userIdResult = sessionStorage.getItem("userId")
    if (userIdResult) return JSON.parse(userIdResult);
    const response = await fetch('/session', {
        headers: {'X-Requested-With': 'XMLHttpRequest'}
    })
    userIdResult = await response.json();
    sessionStorage.setItem("userId", JSON.stringify(userIdResult))
    return userIdResult;
}

let currentPageIndex = 0;
const pageOrder = ['/welcome', '/experiment', '/thanks'];

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

// Load init
init();

// Draw image
const context = canvas.getContext('2d');

const SONG_DURATION = 20;
const IMAGE_FPS = 10;
const BUFFER_SIZE = 150;


document.getElementById("nextButton").addEventListener('click', function(event) {
    setNextButtonDisabled(true);
    set_loading();
    currentPageIndex = (currentPageIndex + 1) % pageOrder.length;
    fetch_content(pageOrder[currentPageIndex], 'mainSection');
});

document.getElementById("stopButton").addEventListener('click', async function (event) {
    setElementHidden("stopButton", true);
    userId = await getUserId();
    requestStop(userId);
    document.getElementById("message").innerText = "The experiment is ending. Please wait a few seconds..."
});

// Start fetching first page
setNextButtonDisabled(true);
setElementHidden("stopButton", true);
// Expose the video tag
setElementHidden("videoSection", false);
fetch_content(pageOrder[currentPageIndex], "mainSection");

