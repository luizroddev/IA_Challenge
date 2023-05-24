// Obtém o elemento do Pokémon por ID
const pokemonElement = document.getElementById("pokemon_label")

// Obtém o elemento do input do tipo file por ID
const uploadPhotoElement = document.getElementById("upload-photo")

// Adiciona o evento de clique ao elemento do Pokémon
pokemonElement.addEventListener("click", function () {
	// Ativa o input do tipo file
	uploadPhotoElement.click()
})
