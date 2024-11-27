// script.js

let allSongs = [];

let surpriseSongsHistory = [];
let currentSurpriseIndex = -1;

let recommendationsHistory = [];
let currentRecommendationsIndex = -1;

async function loadSongs() {
    const response = await fetch('/get_songs');
    const data = await response.json();
    allSongs = data.songs;
    const songsDropdown = document.getElementById('songs');

    allSongs.forEach(song => {
        const option = document.createElement('option');
        option.value = song;
        option.textContent = song;
        songsDropdown.appendChild(option);
    });
}

function filterSongs() {
    const searchValue = document.getElementById('searchBar').value.toLowerCase();
    const songsDropdown = document.getElementById('songs');
    songsDropdown.innerHTML = '';

    allSongs
        .filter(song => song.toLowerCase().includes(searchValue))
        .forEach(song => {
            const option = document.createElement('option');
            option.value = song;
            option.textContent = song;
            songsDropdown.appendChild(option);
        });
}

async function getRecommendations() {
    // Reset the recommendations history
    recommendationsHistory = [];
    currentRecommendationsIndex = -1;
    updateRecommendationsNavigationButtons();
    updateRecommendationsNavigationIcons();

    // Show the Selected Song and Recommendations containers
    document.getElementById('selectedSongContainer').style.display = 'block';
    document.getElementById('recommendationsContainer').style.display = 'block';

    // Hide the Surprise Song container
    document.getElementById('surpriseContainer').style.display = 'none';

    const selectedSong = document.getElementById('songs').value;

    // Fetch the YouTube thumbnail and video link for the selected song
    const selectedVideoContainer = document.getElementById('selectedVideo');
    selectedVideoContainer.innerHTML = ''; // Clear existing video

    const selectedVideoResponse = await fetch('/get_thumbnail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_name: selectedSong })
    });
    const selectedVideoData = await selectedVideoResponse.json();

    // Construct the embed URL if possible
    let selectedEmbedUrl = '';
    if (selectedVideoData.video_url.includes('watch?v=')) {
        selectedEmbedUrl = selectedVideoData.video_url.replace('watch?v=', 'embed/');
    }

    // Display the video or search link for the selected song
    selectedVideoContainer.innerHTML = `
        <div class="song-container">
            <p class="song-name">${selectedSong}</p>
            ${selectedEmbedUrl ? `<iframe src="${selectedEmbedUrl}" frameborder="0" allowfullscreen></iframe>` : ''}
            <div class="audio-and-link">
                <!-- Audio player for the selected song -->
                <audio controls>
                    <source src="/static/music/${encodeURIComponent(selectedSong)}.mp3" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
                <!-- YouTube Icon Link -->
                <a href="${selectedVideoData.video_url}" target="_blank" class="youtube-link">
                    <i class="bi bi-youtube"></i>
                </a>
            </div>
        </div>
    `;

    // Clear the Surprise Me results
    document.getElementById('surprise').innerHTML = '';

    // Fetch recommendations for the selected song
    const response = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_song: selectedSong })
    });
    const data = await response.json();

    if (response.status !== 200) {
        alert(data.error || 'An error occurred.');
        return;
    }

    // Add the recommendations to the history and update index
    recommendationsHistory.push(data.recommendations);
    currentRecommendationsIndex = recommendationsHistory.length - 1;
    updateRecommendationsNavigationButtons();
    updateRecommendationsNavigationIcons();

    // Display the recommendations
    displayRecommendations(data.recommendations);
}

function getAllRecommendedSongs() {
    let songs = [];
    for (let recommendations of recommendationsHistory) {
        for (let rec of recommendations) {
            songs.push(rec.song);
        }
    }
    return songs;
}

async function oneMoreTime() {
    const button = document.getElementById('oneMoreTimeButton');
    const icon = button.querySelector('i');
    // Change the icon to 'fill' version
    icon.className = 'bi bi-disc-fill';

    const selectedSong = document.getElementById('songs').value;

    // Get all previously recommended songs
    const shownSongs = getAllRecommendedSongs();

    // Fetch new recommendations excluding previously shown songs
    const response = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_song: selectedSong, shown_songs: shownSongs })
    });
    const data = await response.json();

    if (response.status !== 200) {
        alert(data.error || 'An error occurred.');
        // Revert the icon back
        icon.className = 'bi bi-disc';
        return;
    }

    // Check if we received any recommendations
    if (data.recommendations.length === 0) {
        alert('No more recommendations available.');
        // Revert the icon back
        icon.className = 'bi bi-disc';
        return;
    }

    // Add the new recommendations to the history and update index
    recommendationsHistory.push(data.recommendations);
    currentRecommendationsIndex = recommendationsHistory.length - 1;
    updateRecommendationsNavigationButtons();
    updateRecommendationsNavigationIcons();

    // Display the new recommendations
    displayRecommendations(data.recommendations);

    // Revert the icon back
    icon.className = 'bi bi-disc';
}

function navigateRecommendations(direction) {
    currentRecommendationsIndex += direction;

    // Ensure index is within bounds
    if (currentRecommendationsIndex < 0) currentRecommendationsIndex = 0;
    if (currentRecommendationsIndex >= recommendationsHistory.length) currentRecommendationsIndex = recommendationsHistory.length - 1;

    // Update navigation buttons
    updateRecommendationsNavigationButtons();
    updateRecommendationsNavigationIcons();

    // Display the recommendations at the current index
    const recommendations = recommendationsHistory[currentRecommendationsIndex];
    displayRecommendations(recommendations);
}

function updateRecommendationsNavigationButtons() {
    const prevButton = document.getElementById('prevRecommendations');
    const nextButton = document.getElementById('nextRecommendations');

    // Disable 'Previous' button if at the first set
    if (currentRecommendationsIndex <= 0) {
        prevButton.disabled = true;
    } else {
        prevButton.disabled = false;
    }

    // Disable 'Next' button if at the last set
    if (currentRecommendationsIndex >= recommendationsHistory.length - 1) {
        nextButton.disabled = true;
    } else {
        nextButton.disabled = false;
    }
}

function updateRecommendationsNavigationIcons() {
    const prevButton = document.getElementById('prevRecommendations');
    const nextButton = document.getElementById('nextRecommendations');

    const prevIcon = prevButton.querySelector('i');
    prevIcon.className = prevButton.disabled ? 'bi bi-arrow-left-circle' : 'bi bi-arrow-left-circle-fill';

    const nextIcon = nextButton.querySelector('i');
    nextIcon.className = nextButton.disabled ? 'bi bi-arrow-right-circle' : 'bi bi-arrow-right-circle-fill';
}

async function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = ''; // Clear existing recommendations

    // Show the recommended songs with embedded videos and audio players
    for (const rec of recommendations) {
        const thumbnailResponse = await fetch('/get_thumbnail', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ song_name: rec.song })
        });

        const thumbnailData = await thumbnailResponse.json();

        // Construct the embed URL if possible
        let embedUrl = '';
        if (thumbnailData.video_url.includes('watch?v=')) {
            embedUrl = thumbnailData.video_url.replace('watch?v=', 'embed/');
        }

        recommendationsDiv.innerHTML += `
            <div class="song-container">
                <p class="song-name">${rec.song}</p>
                ${embedUrl ? `<iframe src="${embedUrl}" frameborder="0" allowfullscreen></iframe>` : ''}
                <div class="audio-and-link">
                    <!-- Audio player for the recommended song -->
                    <audio controls>
                        <source src="/static/music/${encodeURIComponent(rec.song)}.mp3" type="audio/mpeg">
                        Your browser does not support the audio element.
                    </audio>
                    <!-- YouTube Icon Link -->
                    <a href="${thumbnailData.video_url}" target="_blank" class="youtube-link">
                        <i class="bi bi-youtube"></i>
                    </a>
                
        `;
    }
}

async function surpriseMe() {
    // Reset the surprise song history
    surpriseSongsHistory = [];
    currentSurpriseIndex = -1;
    updateSurpriseNavigationButtons();
    updateSurpriseNavigationIcons();

    // Show the Surprise Song container
    document.getElementById('surpriseContainer').style.display = 'block';

    const selectedSong = document.getElementById('songs').value;

    // Clear and hide the Recommendations section
    document.getElementById('recommendations').innerHTML = '';
    document.getElementById('recommendationsContainer').style.display = 'none';

    // Hide the Selected Song section
    document.getElementById('selectedSongContainer').style.display = 'none';

    // Fetch the surprise song (least similar)
    const response = await fetch('/surprise_me', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_song: selectedSong, first: true })
    });
    const data = await response.json();

    if (response.status !== 200) {
        alert(data.error || 'An error occurred.');
        return;
    }

    // Add the song to the history and update index
    surpriseSongsHistory.push(data);
    currentSurpriseIndex = surpriseSongsHistory.length - 1;
    updateSurpriseNavigationButtons();
    updateSurpriseNavigationIcons();

    // Display the surprise song
    displaySurpriseSong(data);
}

async function anotherOne() {
    const button = document.getElementById('anotherOneButton');
    const icon = button.querySelector('i');
    // Change the icon to 'fill' version
    icon.className = 'bi bi-arrow-through-heart-fill';

    const selectedSong = document.getElementById('songs').value;

    // Fetch a new surprise song from the server
    const response = await fetch('/surprise_me', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            seed_song: selectedSong,
            first: false,
            shown_songs: surpriseSongsHistory.map(song => song.song)
        })
    });
    const data = await response.json();

    if (response.status !== 200) {
        alert(data.error || 'An error occurred.');
        // Revert the icon back
        icon.className = 'bi bi-arrow-through-heart';
        return;
    }

    // Add the song to the history and update index
    surpriseSongsHistory.push(data);
    currentSurpriseIndex = surpriseSongsHistory.length - 1;
    updateSurpriseNavigationButtons();
    updateSurpriseNavigationIcons();

    // Display the surprise song
    displaySurpriseSong(data);

    // Revert the icon back
    icon.className = 'bi bi-arrow-through-heart';
}

function navigateSurprise(direction) {
    currentSurpriseIndex += direction;

    // Ensure index is within bounds
    if (currentSurpriseIndex < 0) currentSurpriseIndex = 0;
    if (currentSurpriseIndex >= surpriseSongsHistory.length) currentSurpriseIndex = surpriseSongsHistory.length - 1;

    // Update navigation buttons
    updateSurpriseNavigationButtons();
    updateSurpriseNavigationIcons();

    // Display the song at the current index
    const data = surpriseSongsHistory[currentSurpriseIndex];
    displaySurpriseSong(data);
}

function updateSurpriseNavigationButtons() {
    const prevButton = document.getElementById('prevSurprise');
    const nextButton = document.getElementById('nextSurprise');

    // Disable 'Previous' button if at the first song
    if (currentSurpriseIndex <= 0) {
        prevButton.disabled = true;
    } else {
        prevButton.disabled = false;
    }

    // Disable 'Next' button if at the last song
    if (currentSurpriseIndex >= surpriseSongsHistory.length - 1) {
        nextButton.disabled = true;
    } else {
        nextButton.disabled = false;
    }
}

function updateSurpriseNavigationIcons() {
    const prevButton = document.getElementById('prevSurprise');
    const nextButton = document.getElementById('nextSurprise');

    // Update Previous button icon
    const prevIcon = prevButton.querySelector('i');
    if (prevButton.disabled) {
        prevIcon.className = 'bi bi-arrow-left-circle';
    } else {
        prevIcon.className = 'bi bi-arrow-left-circle-fill';
    }

    // Update Next button icon
    const nextIcon = nextButton.querySelector('i');
    if (nextButton.disabled) {
        nextIcon.className = 'bi bi-arrow-right-circle';
    } else {
        nextIcon.className = 'bi bi-arrow-right-circle-fill';
    }
}

async function displaySurpriseSong(data) {
    const surpriseDiv = document.getElementById('surprise');
    surpriseDiv.innerHTML = '';

    // Fetch thumbnail and video URL
    const thumbnailResponse = await fetch('/get_thumbnail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_name: data.song })
    });

    const thumbnailData = await thumbnailResponse.json();

    // Construct the embed URL if possible
    let embedUrl = '';
    if (thumbnailData.video_url.includes('watch?v=')) {
        embedUrl = thumbnailData.video_url.replace('watch?v=', 'embed/');
    }

    // Display the surprise song
    surpriseDiv.innerHTML = `
        <div class="song-container">
            <p class="song-name">${data.song}</p>
            ${embedUrl ? `<iframe src="${embedUrl}" frameborder="0" allowfullscreen></iframe>` : ''}
            <div class="audio-and-link">
                <!-- Audio player for the surprise song -->
                <audio controls>
                    <source src="/static/music/${encodeURIComponent(data.song)}.mp3" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
                <!-- YouTube Icon Link -->
                <a href="${thumbnailData.video_url}" target="_blank" class="youtube-link">
                    <i class="bi bi-youtube"></i>
                </a>
                
    `;
}

window.onload = loadSongs;
