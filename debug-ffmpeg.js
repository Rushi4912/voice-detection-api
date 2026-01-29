const ffmpeg = require('ffmpeg-static');
const ffprobeLocal = require('@ffprobe-installer/ffprobe');
const fs = require('fs');

console.log('ffmpeg path:', ffmpeg);
console.log('ffprobe path:', ffprobeLocal.path);

if (ffmpeg) {
    console.log('ffmpeg exists:', fs.existsSync(ffmpeg));
} else {
    console.log('ffmpeg is null/undefined');
}

if (ffprobeLocal.path) {
    console.log('ffprobe exists:', fs.existsSync(ffprobeLocal.path));
}
