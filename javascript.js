// this is where all the javascript code will be stored

// function processTransaction(amount, account) {
// 	// Validate transaction
// 	if (amount <= 0) {
// 		console.error('Transaction amount must be greater than zero.');
// 		return false;
// 	}
// 	if (!account.isActive) {
// 		console.error('Account is inactive.');
// 		return false;
// 	}

// 	// Calculate new balance
// 	const newBalance = account.balance + amount;

// 	// Log transaction
// 	console.log(
// 		`Transaction for ${amount} processed. New balance: ${newBalance}.`
// 	);

// 	// Update account balance
// 	account.balance = newBalance;

// 	return true;
// }

function processTransaction(amount, account) {
	// Validate transaction
	if (amount <= 0) {
		console.error('Transaction amount must be greater than zero.');
		return false;
	}
	if (!account.isActive) {
		console.error('Account is inactive.');
		return false;
	}

	// Calculate new balance
	const newBalance = account.balance + amount;

	// Log transaction
	console.log(
		`Transaction for ${amount} processed. New balance: ${newBalance}.`
	);

	// Update account balance
	account.balance = newBalance;

	return true;
}
