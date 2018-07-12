From https://blog.openai.com/requests-for-research-2/.

⭐ Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end. Test the two approaches below:

- Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
- Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

## Files

* [2 bit strings](./xor1.jl)
* [2000 1 to 10 length strings](./xor2.jl)
* [100,000 1 to 50 length strings](./xor3.jl)
