const http = require('http');
const fs = require('fs');
const { execSync } = require('child_process');
const ffmpegPath = require('ffmpeg-static');
const path = require('path');

// Generate 1s silent MP3
console.log('Generating test audio...');
const testFilePath = path.join(__dirname, 'test.mp3');

try {
    if (fs.existsSync(testFilePath)) fs.unlinkSync(testFilePath);

    // Generate 1 second of silence
    execSync(`"${ffmpegPath}" -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -b:a 128k -acodec libmp3lame "${testFilePath}"`, { stdio: 'ignore' });

    const audioBuffer = fs.readFileSync(testFilePath);
    const base64Audio = audioBuffer.toString('base64');
    console.log(`Generated audio size: ${audioBuffer.length} bytes`);

    const postData = JSON.stringify({
        audioBase64: base64Audio,
        audioFormat: "mp3",
        language: "Tamil"
    });

    const options = {
        hostname: 'localhost',
        port: 3000,
        path: '/api/detect',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData),
            'x-api-key': 'master_hackathon_2024' // Key from .env
        }
    };

    console.log('Sending request to Voice Detection API...');

    const req = http.request(options, (res) => {
        console.log(`STATUS: ${res.statusCode}`);
        let data = '';

        res.on('data', (chunk) => {
            data += chunk;
        });

        res.on('end', () => {
            console.log('Response received:');
            try {
                const jsonResponse = JSON.parse(data);
                console.log(JSON.stringify(jsonResponse, null, 2));

                if (res.statusCode !== 200) {
                    console.log('\n❌ Request failed with status', res.statusCode);
                    return;
                }

                // Verify fields
                const requiredFields = ['status', 'language', 'classification', 'confidenceScore', 'explanation', 'analysis'];
                const missing = requiredFields.filter(field => !jsonResponse.hasOwnProperty(field));

                if (missing.length === 0) {
                    console.log('\n✅ format verification PASSED: All required fields are present.');
                    if (jsonResponse.status === 'success') {
                        console.log('✅ Status is success');
                    } else {
                        console.log('❌ Status is NOT success');
                    }
                } else {
                    console.log('\n❌ Format verification FAILED. Missing fields:', missing);
                }

            } catch (e) {
                console.log('Response body:', data);
                console.error('Failed to parse JSON response:', e);
            }

            // Cleanup
            if (fs.existsSync(testFilePath)) fs.unlinkSync(testFilePath);
        });
    });

    req.on('error', (e) => {
        console.error(`problem with request: ${e.message}`);
        if (fs.existsSync(testFilePath)) fs.unlinkSync(testFilePath);
    });

    req.write(postData);
    req.end();

} catch (err) {
    console.error('Failed to generate test audio:', err);
}
