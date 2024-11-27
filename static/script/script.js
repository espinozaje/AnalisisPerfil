  // Función para abrir el modal con la imagen que se hizo clic
  function openModal(imgSrc) {
    var modal = document.getElementById("image-modal");
    var modalImg = document.getElementById("modal-img");
    modal.style.display = "block";  // Mostrar el modal
    modalImg.src = imgSrc;  // Establecer la imagen en el modal
}

// Función para cerrar el modal
function closeModal() {
    var modal = document.getElementById("image-modal");
    modal.style.display = "none";  // Ocultar el modal
}