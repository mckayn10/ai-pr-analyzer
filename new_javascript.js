export function displayCart() {
	const cart = JSON.parse(localStorage.getItem('cart')) || [];
	updateCartDisplay(cart);
}
