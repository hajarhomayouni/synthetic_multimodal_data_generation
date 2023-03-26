# Synthetic Multimodal Data Generation
---
## Approach
* Text-to-text, text-to-image
* GPT2 model for text-to-text
## Related work
Pre-trained model
[GPT2-SIMPLE](https://github.com/minimaxir/gpt-2-simple )
[AI-GEN](https://github.com/minimaxir/aitextgen)
## Performace Evaluate
- Loss function
- Preplexity
  - math.exp(loss/ len(token_input))
- BLEU
  - Rely on the prefix input and origin text
- the following paper indicate that lower preplexity has better performance
  - [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)