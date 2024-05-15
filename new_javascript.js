export function displayCart() {
	const cart = JSON.parse(localStorage.getItem('cart')) || [];
	updateCartDisplay(cart);
}

function updateCartDisplay(cart) {
	const cartDisplay = document.getElementById('cart');
	cartDisplay.innerHTML = ''; // Clear current display
	cart.forEach((item) => {
		const productElement = document.createElement('div');
		productElement.textContent = `${item.name}: ${item.quantity}`;
		cartDisplay.appendChild(productElement);
	});
	console.log('Cart display updated.');
}
