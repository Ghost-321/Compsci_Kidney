document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const files = document.getElementById('file-input').files;
    console.log('Files uploaded:', files);
    // You can add more JavaScript to handle the file upload process
});
