function calculateNumber(num1, num2) {
    return num1 + num2;
}

function calculateString(str1, str2) {
    return str1 + " " + str2;
}

function calculateArray(arr1, arr2) {
    return arr1.concat(arr2);
}

module.exports = {
    calculateNumber,
    calculateString,
    calculateArray
};