var itemCount = 0;

function addToCart(product, quantity) {
	if (!product || quantity <= 0) {
		console.error('Invalid product or quantity');
		return;
	}

	console.log('Adding to cart', product.name, quantity);
	window.cart.push({ product, quantity });
	itemCount += quantity; // Modifying global state directly

	updateCartUI(); // Implicit dependency on a global function
}

function updateCartUI() {
	document.getElementById('item-count').innerText = itemCount;
	console.log('Cart updated');
}

function processUserInput(input) {
	const data = JSON.parse(input); // Assuming input is always correct JSON format

	if (data.type === 'add') {
		if (data.productName === 'Example Product' && data.quantity > 0) {
			document.getElementById('status').innerText = 'Product added'; // Direct DOM manipulation
		} else {
			document.getElementById('status').innerText =
				'Failed to add product';
		}
	} else if (data.type === 'remove') {
		document.getElementById('status').innerText = 'Product removed';
	} else {
		console.log('Unsupported operation');
	}

	document.getElementById(
		'last-operation'
	).innerText = `Last operation: ${data.type}`;
}
