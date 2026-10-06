<!-- ======== DL 1-5 ======== -->
# 1. Multi-Layer Perceptron (MLP)

## Why we care

An MLP is a flexible model for learning relationships between inputs and outcomes. For example, it can learn how age, education, region, and media exposure relate to a person’s probability of supporting a political party.

It is built from artificial **neurons**, which are small mathematical units. With enough neurons and suitable activation functions, an MLP can approximate a very large class of continuous relationships—an idea known as the **universal approximation theorem** (Cybenko, 1989).

## One-sentence intuition

An MLP learns increasingly complicated combinations of simple numerical signals.

## How it works

Suppose the input is a vector of numbers, such as:

- Age
- State
- Education level
- Urban or rural location
- Survey responses

Each neuron:

1. Multiplies each input by a **weight**, representing how important that input currently seems.
2. Adds the results together.
3. Adds a **bias**, which shifts the neuron’s response.
4. Applies an **activation function**, which introduces nonlinearity.

Without nonlinear activation functions, many layers would behave like one large linear equation and could not learn complicated patterns.

Common activations include:

- **ReLU:** outputs zero for negative values and keeps positive values.
- **GELU:** smoothly scales values rather than abruptly cutting off negatives.
- **Softmax:** converts final scores into probabilities that add up to 1, useful for choosing among parties or candidates.

During training, the network compares its prediction with the known answer and adjusts its weights and biases.

## Everyday analogy

Imagine a committee evaluating a restaurant. One member considers price, another taste, another location, and another service. Each member gives some factors more importance, adds a personal “starting preference,” and produces a judgment. Several layers of committees can combine these judgments into a final recommendation.

## On our Mexican elections

Each person or municipality could be one example. The inputs might include demographics, economic indicators, state, polling responses, and social-media behavior. The MLP could output:

- The predicted party choice
- Probabilities for each party
- The likelihood that someone votes

For categorical variables such as state, we would usually encode them numerically, often with one-hot encoding or learned embeddings. The model would learn combinations such as “young + urban + university-educated + particular region,” rather than relying only on one variable at a time.

## Key paper

Cybenko, G. (1989), “Approximation by Superpositions of a Sigmoidal Function.”

# 2. PyTorch and Automatic Differentiation

## Why we care

A neural network may contain thousands or millions of adjustable numbers. Calculating how every number should change by hand would be extremely slow. **PyTorch** provides tools for representing data, building neural networks, calculating gradients, and training models.

## One-sentence intuition

Automatic differentiation tells us how much each parameter contributed to the model’s error, so the model knows how to improve.

## How it works

A **tensor** is a container of numbers: it may be a single number, a list, a table, or a higher-dimensional array.

PyTorch records operations performed on tensors in a **computational graph**. For example:

\[
\text{inputs} \rightarrow \text{weighted sum} \rightarrow \text{activation} \rightarrow \text{prediction} \rightarrow \text{loss}
\]

The **loss** measures how wrong the prediction is.

Using the chain rule from calculus, PyTorch performs **backpropagation**. It works backward through the graph and computes a **gradient** for each parameter. A gradient indicates the direction in which changing a parameter would increase the loss.

The optimizer then updates parameters in the opposite direction:

\[
\text{new parameter}
=
\text{old parameter}
-
\text{learning rate} \times \text{gradient}
\]

- **Gradient descent** uses this principle repeatedly.
- **Stochastic gradient descent (SGD)** estimates the gradient using a small random batch of examples.
- **Adam** adapts the step size for each parameter using running information about recent gradients.

This process repeats over many batches and **epochs**, where one epoch means passing through the training data once.

## Everyday analogy

Suppose you are learning to throw a ball at a target. After each throw, someone tells you whether you were too far left, too far right, too high, or too low. You use that feedback to adjust your next throw. Automatic differentiation provides similar directional feedback for every model parameter.

## On our Mexican elections

The dataset would be stored as tensors. One tensor might contain voter features, and another might contain the observed election outcome. PyTorch would:

1. Pass the features through the model.
2. Produce predicted party probabilities.
3. Calculate a loss against the actual result.
4. Compute gradients.
5. Use SGD or Adam to update the model.

The data should be divided into training, validation, and test sets so that performance measures whether the model generalizes to new voters, municipalities, or elections.

## Key paper

Paszke, A., et al. (2019), “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” The earlier automatic-differentiation work is commonly cited as Paszke et al. (2017).

# 3. Regularization

## Why we care

A model can **overfit**: it may memorize peculiarities of the training data instead of learning patterns that apply to new cases. Regularization discourages unnecessarily complicated solutions and usually improves generalization.

The central issue is the **bias-variance tradeoff**:

- High bias: the model is too simple and misses real patterns.
- High variance: the model is too sensitive to the particular training sample.

Regularization usually accepts a little more bias to reduce variance.

## One-sentence intuition

Regularization teaches a model to learn the important pattern without memorizing every accident in the data.

## How it works

### L2 weight decay

L2 regularization adds a penalty for large weights to the loss:

\[
\text{total loss}
=
\text{prediction loss}
+
\lambda \sum_i w_i^2
\]

The model is encouraged to use smaller, smoother parameter values rather than relying excessively on a few very large weights. In neural-network optimizers, this is often called **weight decay**.

### Dropout

During training, dropout randomly turns off some neurons. The network must therefore avoid depending on any single neuron and learn more distributed representations. At evaluation time, all neurons are used, with appropriate scaling.

### Early stopping

Training is stopped when performance on a validation set stops improving. This prevents the model from continuing to fit noise in the training data.

## Everyday analogy

Imagine studying for an exam. Memorizing the exact wording of practice questions may give excellent practice-test results but poor performance on new questions. A better strategy is to learn the underlying ideas, use several study methods, and stop practicing a topic once improvement has ended.

## On our Mexican elections

Election data may contain accidental correlations:

- A specific survey wave
- A temporary news event
- A small municipality with unusual results
- A demographic group overrepresented in the sample

Regularization can help prevent the model from treating these quirks as permanent political laws. We could use L2 weight decay, dropout in the MLP, and early stopping based on validation performance.

Validation splits must respect the prediction goal. For example, if we want to predict a future election, a time-based split is more realistic than randomly mixing past and future observations.

## Key paper

Srivastava, N., et al. (2014), “Dropout: A Simple Way to Prevent Neural Networks from Overfitting.”

# 4. Convolutional Neural Network (CNN)

## Why we care

A CNN is especially useful for data with local structure, such as images, maps, audio spectrograms, or spatial grids. Nearby elements often have related meanings, and the same pattern may appear in different locations.

CNNs exploit this structure while using far fewer parameters than a fully connected network.

## One-sentence intuition

A CNN scans for small, reusable patterns and combines them into larger patterns.

## How it works

A **kernel**, or filter, is a small grid of learnable numbers. The CNN slides this kernel across an input, multiplying and adding local values at each position. This operation is called a **convolution**.

Early filters may learn simple patterns. In an image, they might detect edges; in a map, they might detect local geographic arrangements. Later layers combine these patterns into more complex structures.

Important concepts include:

- **Weight sharing:** the same filter is used at every location, so the model can detect the same pattern wherever it appears.
- **Receptive field:** the part of the original input that can influence a particular neuron. Deeper layers generally have larger receptive fields.
- **Pooling:** an operation such as taking the maximum or average within a small region. It reduces resolution and can make representations less sensitive to small shifts.

CNNs are not automatically appropriate for ordinary tabular election data. They make sense when the data has meaningful spatial or ordered structure.

## Everyday analogy

Imagine scanning a page with a small magnifying glass. First, you look for simple shapes such as lines. Then you combine lines into letters, letters into words, and words into larger ideas. The same magnifying glass can search for a shape anywhere on the page.

## On our Mexican elections

A CNN could be useful in several election-related settings:

- A grid of municipality-level voting results
- A geographic map of voting patterns
- Satellite images linked to municipalities
- Ordered sequences of polling or media activity
- Text converted into suitable numerical representations

For a geographic CNN, nearby municipalities would be arranged in a grid or another spatial representation. Filters could learn local patterns such as neighboring regions with similar turnout or party support.

However, a simple grid may incorrectly treat two municipalities as neighbors merely because they are placed nearby in the grid. For irregular geographic networks, graph neural networks or models using explicit geographic coordinates may be more appropriate.

## Key paper

LeCun, Y., Bengio, Y., and Hinton, G. (2015), “Deep Learning.”

# 5. Transfer Learning

## Why we care

Large models often require enormous datasets and substantial computing resources. **Transfer learning** reuses knowledge learned from one task or dataset to help with another, usually related, task.

This is valuable for Mexican election research because labeled political data may be limited, especially for particular states, demographic groups, or election years.

## One-sentence intuition

Transfer learning starts with a model that already knows useful general patterns and adapts it to the specific problem.

## How it works

### Pre-training

The model first learns from a large source dataset or task. For example, a language model may learn general language patterns from large collections of text.

### Fine-tuning

The model is then trained further on the Mexican elections dataset so that it learns election-specific relationships.

### Freezing

Some early layers can be **frozen**, meaning their parameters are not updated. This preserves general knowledge while only later layers adapt. Alternatively, all layers can be fine-tuned, often with a smaller learning rate.

### Domain shift

The source and target data may differ. This is called **domain shift**. For example:

- General Spanish text may differ from Mexican political language.
- National polling may differ from a particular state.
- One election year may differ from another.
- Online users may differ from the entire voting population.

Transfer learning helps, but it does not remove these differences. The model must be evaluated on data representing the intended prediction setting.

## Everyday analogy

A student who already speaks Spanish can learn Italian more easily than someone who has never learned another language. Existing knowledge helps, but the student still needs to learn Italian-specific vocabulary, grammar, and pronunciation.

## On our Mexican elections

Possible examples include:

- Starting with a Spanish-language language model and fine-tuning it on Mexican campaign speeches, news, or social-media posts.
- Starting with a model trained on broad demographic or survey data and adapting it to Mexican election outcomes.
- Starting with a model trained on previous Mexican elections and fine-tuning it for a new election year.

We should compare transfer learning against training from scratch. We should also check whether the source data contains information unavailable at prediction time, such as future election results, because that would create leakage and misleadingly strong performance.

## Key paper

Pan, S. J., and Yang, Q. (2010), “A Survey on Transfer Learning.”
<!-- ======== DL 6-10 ======== -->
## 6. Autoencoder (AE) and Variational Autoencoder (VAE)

### Why we care

An **autoencoder** learns to compress complicated data into a smaller representation and then reconstruct it. This is useful for visualization, noise removal, anomaly detection, and discovering hidden patterns.

A **VAE** adds probability and randomness. Instead of assigning each example one exact compressed code, it learns a smooth region of possible codes. This makes it possible to generate new, realistic examples.

### One-sentence intuition

An autoencoder learns to summarize data; a VAE learns a smooth, organized space of summaries from which new data can be generated.

### How it works

An autoencoder has two parts:

- The **encoder** converts an input \(x\) into a smaller representation \(z\).
- The **decoder** converts \(z\) back into a reconstruction of \(x\).

The narrow middle is the **bottleneck**. Because it has limited space, the model must preserve the most important information.

This connects to the **manifold hypothesis**: although data may have many measured variables, meaningful examples often lie near a lower-dimensional shape, or *manifold*. For example, millions of possible pixel arrangements may represent only a small number of recognizable faces.

A VAE changes the encoder’s output. Rather than producing one code, it produces a probability distribution, usually described by a mean \(\mu\) and standard deviation \(\sigma\). The decoder samples a latent code \(z\) from that distribution.

The VAE maximizes the **ELBO**—the evidence lower bound:

\[
\text{ELBO}
=
\mathbb{E}_{q(z|x)}[\log p(x|z)]
-
D_{\mathrm{KL}}(q(z|x)\|p(z)).
\]

The first term rewards accurate reconstruction. The second term is the **KL divergence**, which measures how different the learned latent distribution is from a simple prior, often a standard normal distribution. This regularizes the latent space so nearby codes produce similar outputs.

Sampling seems to block ordinary backpropagation, but the **reparameterization trick** rewrites sampling as:

\[
z=\mu+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
\]

The randomness is placed in \(\epsilon\), while \(\mu\) and \(\sigma\) remain differentiable and can be learned.

### Everyday analogy

Imagine reducing a complicated meal to a recipe. An autoencoder writes a compact recipe and uses it to recreate the meal. A VAE does more: it learns a smooth cookbook in which similar recipes are near one another, allowing it to invent a plausible new recipe between existing ones.

### On our Mexican elections

The input could contain municipality-level variables such as turnout, vote share for each party, invalid votes, population, urbanization, income, education, and previous-election results.

- An AE could compress these variables into two or three latent dimensions for visualization.
- Similar municipalities might cluster together according to electoral and socioeconomic patterns.
- A VAE could generate synthetic municipality profiles for simulation or privacy-preserving experimentation.
- Reconstruction error could identify unusual municipalities, such as places with unexpectedly large turnout or vote changes.

A latent dimension should not automatically be called “political ideology” or “corruption.” It is a learned mathematical feature whose meaning must be investigated.

### Key paper

Kingma and Welling (2013), *Auto-Encoding Variational Bayes*.

---

## 7. Transformer

### Why we care

Language depends on relationships between words that may be far apart. In “The candidate who visited Oaxaca gave a speech,” the verb “gave” is connected to “candidate,” not merely to the nearest word.

Transformers model these relationships efficiently and are the foundation of modern language models, translation systems, summarizers, and many speech and vision systems.

### One-sentence intuition

A Transformer lets every word examine the other words and decide which ones matter most for understanding it.

### How it works

Each token—such as a word or word fragment—is converted into a numerical vector called an **embedding**.

For each token, the model creates:

- A **query**: what information it is looking for.
- A **key**: what kind of information it offers.
- A **value**: the information it will provide.

The core operation is **scaled dot-product attention**:

\[
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

In plain language, the model compares queries with keys, turns the comparison scores into weights, and combines the values accordingly. Scaling prevents the scores from becoming excessively large.

**Multi-head attention** runs several attention mechanisms in parallel. One head might focus on grammar, another on names, and another on long-distance relationships.

Because attention alone does not know word order, the model adds **positional encodings**. These tell it whether a token comes first, second, or later.

In a language-generating Transformer, **causal masking** prevents the model from looking at future tokens. When predicting the next word, it may use the past but not the answer it is supposed to predict.

Transformers also use:

- **Residual connections**, which add a layer’s input to its output and help information and gradients flow through many layers.
- **LayerNorm**, which stabilizes the values inside each example and makes training more reliable.

### Everyday analogy

Imagine a student answering a question while consulting a classroom. The student asks different classmates different questions, listens more carefully to the most relevant answers, and remembers where each classmate is sitting. Residual connections let the student keep the original notes, while LayerNorm keeps the notes organized.

### On our Mexican elections

A Transformer could process a sequence such as:

“Municipality: Puebla; turnout: 61%; PAN: 38%; PRI: 14%; Morena: 42%; previous election: Morena: 35%.”

Attention could connect:

- Current vote share to previous-election results.
- Turnout to invalid votes.
- Municipality characteristics to electoral changes.
- Words in newspaper reports to locations, parties, and events.

For tabular data, each variable can be represented as a token or embedding. For electoral news, the Transformer can analyze text about campaigns, coalitions, violence, or electoral institutions.

Causal masking would be useful for forecasting future results using only information available before election day. It would be inappropriate to let the model use post-election information when evaluating a historical forecast.

### Key paper

Vaswani et al. (2017), *Attention Is All You Need*.[1]

---

## 8. Automatic Speech Recognition (ASR)

### Why we care

**Automatic Speech Recognition** converts spoken language into written text. It enables transcriptions of interviews, campaign speeches, debates, citizen complaints, and radio broadcasts.

### One-sentence intuition

ASR turns a changing sound wave into a sequence of words by recognizing patterns in the sound and predicting the most likely transcript.

### How it works

A microphone records air-pressure changes as a waveform. The waveform is transformed into a **spectrogram**, which shows how much energy exists at different frequencies over time.

A **Mel spectrogram** uses the Mel frequency scale, which represents frequencies roughly according to human hearing. It is usually computed by:

1. Dividing the audio into short overlapping time windows.
2. Applying a frequency analysis to each window.
3. Grouping frequencies into Mel-scaled bands.
4. Recording the energy in each band.

The result is an image-like matrix: time on one axis, frequency on the other, and intensity represented by brightness or numerical value.

An encoder processes this sequence of acoustic features and creates useful internal representations. A decoder then generates text one token at a time. Modern systems may use attention or a Transformer, while older systems often used recurrent networks.

The most common evaluation measure is **word error rate (WER)**:

\[
\text{WER}
=
\frac{S+D+I}{N},
\]

where:

- \(S\) = substitutions,
- \(D\) = deletions,
- \(I\) = insertions,
- \(N\) = number of words in the correct transcript.

Lower WER is better. A system that confuses “Morena” with another word makes a substitution; one that misses “not” makes a deletion.

### Everyday analogy

ASR is like listening to a muffled radio broadcast. First, you examine the sound pattern. Then you use context: if the speaker says “the governor of…,” a place name is more likely to follow than an unrelated word.

### On our Mexican elections

ASR could transcribe:

- Candidate speeches.
- Electoral debates.
- Interviews with voters.
- Local radio programs.
- Election-monitoring hotlines.
- Indigenous-language or Spanish-language political content.

The Mexican setting creates important challenges: accents, background noise, code-switching, regional vocabulary, names of municipalities, and Indigenous languages. WER should therefore be reported separately for regions, speakers, languages, and politically important terms.

A low overall WER can still hide serious errors if the system frequently misrecognizes candidate names, party names, or municipality names.

### Key paper

Graves et al. (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*.

---

## 9. LSTM / GRU / seq2seq

### Why we care

Many problems involve sequences: election results across years, monthly turnout, speech over time, or a sentence being translated word by word. Ordinary neural networks do not naturally remember information over long sequences.

**LSTMs** and **GRUs** were designed to maintain useful information over time while forgetting irrelevant information. **Seq2seq** models use one sequence-processing network to create another sequence.

### One-sentence intuition

LSTMs and GRUs learn what to remember, what to forget, and what to output at each step in a sequence.

### How it works

A basic recurrent neural network processes one time step at a time. It receives the current input \(x_t\) and its previous hidden state \(h_{t-1}\), then creates a new hidden state \(h_t\).

The problem is that information from far earlier steps can disappear during training. This is called the vanishing-gradient problem.

An **LSTM** uses a memory cell and gates:

- The **forget gate** decides what old information to discard.
- The **input gate** decides what new information to store.
- The **output gate** decides what information to reveal.

A **GRU** is a simpler alternative. It uses update and reset gates and does not maintain a separate memory cell. It often trains faster while still handling longer dependencies better than a basic RNN.

A **seq2seq** system has:

- An **encoder**, which reads the input sequence.
- A **decoder**, which generates the output sequence.

For translation, the encoder reads a sentence and the decoder produces the translated sentence. Attention can help the decoder consult different parts of the input rather than relying on one fixed summary.

During training, **teacher forcing** gives the decoder the correct previous output instead of its own previous prediction. This makes learning easier, but at test time the correct previous answer is unavailable. The resulting mismatch is called exposure bias.

### Everyday analogy

Think of a secretary following a long meeting. The secretary keeps a notebook, crosses out outdated information, writes down important new facts, and decides which notes to mention in the final report. Teacher forcing is like practicing with an answer key; real life requires continuing even after making a mistake.

### On our Mexican elections

An LSTM or GRU could model:

- A municipality’s vote shares across several elections.
- Monthly turnout or registration patterns.
- Campaign events over time.
- Sequences of political news.
- Speech audio features.

For example, the model could use election results from 2000–2018 to predict a municipality’s 2021 party vote share. The hidden state would represent the model’s learned summary of electoral history.

A seq2seq model could convert a sequence of campaign events into a sequence of predicted outcomes, or summarize a sequence of local news reports. Care is needed: historical sequences must be ordered correctly, and training data must not include future information.

### Key paper

Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*.

---

## 10. Large Language Models (LLMs)

### Why we care

A **large language model** learns statistical patterns in enormous collections of text. It can generate, summarize, classify, translate, and answer questions. For electoral research, it can help organize campaign documents, extract claims, code political themes, and search large archives.

### One-sentence intuition

An LLM repeatedly predicts what token is most likely to come next, and its ability to do this at enormous scale produces surprisingly broad language skills.

### How it works

Text is divided into **tokens**, which may be words, parts of words, punctuation marks, or symbols.

During training, the model sees a sequence such as:

> “Voters in the municipality…”

and tries to predict the next token. It compares its prediction with the actual next token and adjusts its parameters to reduce the error. Repeating this over huge datasets teaches the model grammar, facts, writing patterns, and associations.

Modern LLMs generally use Transformer architectures. **Scale** matters in several ways:

- More parameters provide more capacity.
- More training data exposes the model to more patterns.
- More computation allows more optimization.

However, scale does not guarantee truth. A **hallucination** is a fluent but unsupported or false output. The model is trained to produce likely text, not to directly verify every statement against reality. It may therefore invent citations, confuse similarly named municipalities, or report an outdated election result confidently.

### Everyday analogy

An LLM is like an extremely well-read student who has practiced completing billions of sentences. The student is excellent at producing plausible language but may sometimes confidently repeat a rumor or fill in a missing fact incorrectly.

### On our Mexican elections

An LLM could assist with:

- Summarizing party platforms.
- Extracting candidate names, offices, promises, and locations.
- Classifying campaign messages by topic.
- Comparing speeches across parties or regions.
- Translating or organizing Spanish-language documents.
- Generating code for electoral analysis.

It should not be treated as an unquestioned source of election facts. Every claim about vote totals, candidates, laws, or dates should be checked against official electoral records or carefully cited documents. Evaluation should include accuracy by state, municipality, party, language, and election year.

Sensitive conclusions—such as inferring an individual voter’s political preference—should not be made merely because a model detects patterns in text or demographic data.

### Key paper

Brown et al. (2020), *Language Models are Few-Shot Learners*.
<!-- ======== DL/NLP 11-15 ======== -->
# 11. Retrieval-Augmented Generation (RAG)

### Why we care

A language model may produce fluent answers while relying on incomplete or outdated internal knowledge. **RAG** gives it access to an external collection of documents, allowing answers to be based on specific evidence rather than memory alone. This matters especially for elections, where dates, rules, candidates, and results must be traceable.

### One-sentence intuition

**RAG is like asking a knowledgeable writer to first look up relevant documents and then answer using what they found.**

### How it works

1. A user asks a question, such as “How did turnout change between two Mexican elections?”
2. The system converts the question into a numerical representation called an **embedding**. Similar meanings have similar numerical representations.
3. It converts stored documents—such as election reports, survey descriptions, or news articles—into embeddings too.
4. A **dense retriever** searches for documents whose embeddings are close to the question’s embedding, even when they do not use exactly the same words.
5. The language model receives the question plus the retrieved passages and generates an answer.
6. We check **faithfulness**: whether each important claim is actually supported by the retrieved evidence.

RAG can still be wrong if retrieval finds poor documents or if the model ignores the evidence. Therefore, answers should include citations, confidence scores, or links to the exact supporting records.

### Everyday analogy

Imagine writing a school report. Instead of trusting your memory, you search the library for the most relevant pages, read them, and write an explanation based on those pages. The library search is retrieval; your report is generation.

### On our Mexican elections

We could store:

- INE reports and official election results.
- Polling questionnaires and methodological notes.
- Candidate speeches and party platforms.
- Municipality-level turnout, vote shares, and demographic statistics.

A user could ask, “Which states had the largest turnout decline, and what explanations were proposed?” RAG would retrieve the relevant tables and reports, then produce an evidence-grounded answer. It should distinguish official facts from interpretations and never invent missing election data.

### Key paper

**Lewis et al. (2020)** introduced a widely used RAG architecture combining a neural retriever with a text-generation model and an external dense document index.[13]

# 12. Agentic Workflows

### Why we care

A normal language model usually responds in one step. An **agentic workflow** lets a model plan several steps, use tools, inspect results, and revise its work. This is useful when an electoral question requires calculations, database searches, document reading, and validation.

### One-sentence intuition

**An agent is a language model that can decide what action to take next instead of only writing an answer.**

### How it works

A typical workflow is:

1. Interpret the user’s question.
2. Break it into smaller tasks.
3. Choose a tool—for example, a database query, calculator, search system, or statistical program.
4. Execute the tool.
5. Examine the result.
6. Decide whether another step is needed.
7. Produce an answer with an **audit trail**.

An audit trail records the question, tools used, inputs, outputs, transformations, and final reasoning. The agent should have safeguards: restricted permissions, human approval for sensitive actions, and checks against unsupported claims.

### Everyday analogy

A research assistant receives a difficult assignment. They make a plan, look up sources, calculate numbers, ask a colleague to verify the result, and keep notes describing every step.

### On our Mexican elections

Suppose someone asks:

> “Did turnout fall more in municipalities with higher poverty between 2018 and 2024?”

An agent could:

- Query the election database.
- Join election results with poverty statistics.
- Calculate turnout changes.
- Run a correlation or regression.
- Check for missing values and unusual municipalities.
- Create a chart.
- Report the method and limitations.

The audit trail would show exactly which datasets, variables, filters, and formulas were used. The agent must not treat correlation as proof that poverty caused turnout changes.

### Key paper

**Yao et al. (2023)** proposed ReAct, a method in which language models alternate between reasoning and actions such as searching or using tools.

# 13. Generative Adversarial Networks (GANs) and Diffusion Models

### Why we care

Election datasets may contain missing regions, rare demographic combinations, or too few examples for safe experimentation. Generative models can create **synthetic** observations or simulate counterfactual situations—for example, estimating what a voting pattern might look like under a changed turnout assumption.

These simulations can help test models, but synthetic data are not automatically true. They must be labeled clearly and evaluated for bias and realism.

### One-sentence intuition

**GANs learn through a competition between a faker and a detective, while diffusion models learn to reconstruct realistic data from gradually added noise.**

### How it works

#### GANs

A GAN contains:

- A **generator**, which creates synthetic data.
- A **discriminator**, which tries to distinguish real data from generated data.

The generator improves by trying to fool the discriminator, while the discriminator improves by detecting false examples. Over time, the generator may learn patterns resembling the training data.

#### Diffusion models

A diffusion model:

1. Gradually adds random noise to real examples during training.
2. Learns how to reverse that process.
3. Starts with noise and repeatedly removes it to create a new example.

For tabular election data, the model might generate combinations of turnout, demographic variables, and vote shares, provided it is designed for structured data rather than images.

#### Counterfactual simulation

A **counterfactual** asks, “What might have happened if something had been different?” Generative models can create plausible scenarios, but they do not discover causal truth by themselves. Causal assumptions and statistical validation are still necessary.

### Everyday analogy

A GAN is like an art student and an art critic competing: the student produces paintings, and the critic identifies fakes. A diffusion model is like restoring a photograph that has been covered with increasing amounts of fog, learning how to remove the fog step by step.

### On our Mexican elections

A model could simulate synthetic municipality-level records resembling the observed data, such as:

- Turnout by age and urbanization.
- Vote shares by party.
- Regional demographic patterns.
- Possible missing responses in a voter survey.

For a counterfactual, we might ask: “What turnout patterns are plausible if youth participation increased by five percentage points?” The output should be presented as a range of modeled scenarios, not as an actual prediction or historical fact. We should test whether generated data preserve real distributions, correlations, regional differences, and privacy protections.

### Key paper

For GANs: **Goodfellow et al. (2014)** introduced the generator–discriminator framework. For diffusion models: **Ho et al. (2020)** introduced a highly influential denoising diffusion approach.

# 14. State-Space Models (SSM)

### Why we care

Election data often have an order: years, months, polling waves, or sequential campaign events. We need models that can remember relevant past information while processing long sequences efficiently.

**State-Space Models** summarize the past in a compact hidden state. Modern models such as **Mamba** use selective state updates, while models such as **TSMixer** mix information across time and variables for forecasting.

### One-sentence intuition

**An SSM reads a sequence one step at a time, continuously updating a compact memory of what has happened so far.**

### How it works

At each time step, an SSM has:

- An observed input, such as turnout in a month.
- A hidden state, representing useful accumulated information.
- An update rule that changes the state.
- An output produced from the current state.

A simplified recurrence is:

\[
s_t = f(s_{t-1}, x_t)
\]

\[
y_t = g(s_t)
\]

Here, \(x_t\) is the current observation, \(s_t\) is the memory, and \(y_t\) is the output.

Unlike standard self-attention, which directly compares many positions with one another, a basic SSM processes information through a recurrence. This can be more efficient for very long sequences. Mamba adds input-dependent, or **selective**, updates so the model can decide what to remember and what to forget. TSMixer is a time-series architecture that mixes information across time points and feature dimensions.

### Everyday analogy

Imagine carrying a small notebook while following a long election campaign. At each event, you update the notebook with the most important information. You do not keep every detail visible, but you maintain a useful summary.

### On our Mexican elections

We could use an SSM to model:

- Monthly polling trends.
- Turnout across successive elections.
- Daily campaign events and social-media indicators.
- Regional changes in party support.
- Economic indicators leading up to election day.

For example, the hidden state might summarize recent momentum, persistent regional differences, and a temporary scandal. A model could forecast the next polling wave or identify periods when voting intentions changed sharply.

The main caution is that a compact memory can lose important details. We should compare SSM forecasts with simpler baselines and attention-based models, and avoid interpreting the hidden state as a direct political explanation.

### Key paper

**Gu and Dao (2023)** introduced Mamba, a selective state-space model designed for efficient sequence modeling with long contexts. **Ni et al. (2023)** introduced TSMixer for time-series forecasting.

# 15. Bag-of-Words (BoW) and TF-IDF

### Why we care

Before using complex language models, we need simple, understandable ways to turn text into numbers. **Bag-of-Words** and **TF-IDF** are foundational methods for classifying, searching, and comparing documents.

They are useful because they are fast and interpretable, even though they do not understand word order or deeper meaning.

### One-sentence intuition

**BoW counts which words appear, while TF-IDF gives more importance to words that are frequent in one document but uncommon across the whole collection.**

### How it works

#### Bag-of-Words

Suppose a document says:

> “Turnout increased in Puebla.”

BoW creates a vocabulary such as:

- turnout
- increased
- Puebla
- candidate
- migration

It represents the document as numbers indicating how often each word appears. Word order is mostly discarded, so “the candidate defeated the party” and “the party defeated the candidate” may look similar.

#### TF-IDF

TF-IDF has two components:

- **Term frequency (TF):** how often a word appears in a particular document.
- **Inverse document frequency (IDF):** how uncommon that word is across all documents.

A common form is:

\[
\mathrm{TFIDF}(w,d)=\mathrm{TF}(w,d)\times \log\left(\frac{N}{\mathrm{DF}(w)}\right)
\]

where \(N\) is the number of documents and \(\mathrm{DF}(w)\) is the number of documents containing word \(w\).

A common word such as “election” may receive a low score because it appears everywhere. A distinctive term such as a municipality name or a specific policy issue may receive a higher score.

### Everyday analogy

Imagine sorting newspaper articles using word magnets. If every article contains “Mexico” and “election,” those magnets tell you little. A rare magnet such as “water scarcity” may help identify what makes one article different.

### On our Mexican elections

We could apply BoW or TF-IDF to:

- Candidate speeches.
- Party platforms.
- News articles.
- Voter survey responses.
- Social-media posts.
- Election-law documents.

Possible tasks include:

- Finding documents about migration, security, or corruption.
- Classifying speeches by topic.
- Comparing party platforms.
- Measuring whether campaign language changed over time.
- Searching for municipalities, candidates, or policy terms.

The method is transparent: we can inspect which words drive a classification. However, it may miss synonyms, sarcasm, negation, spelling variation, and context. “Support the candidate” and “Do not support the candidate” may appear similar unless the model is designed to handle negation.

### Key paper

**Salton and Buckley (1988)** described influential weighting methods for information retrieval, including TF-IDF-style term weighting.
<!-- ======== NLP 16-20 ======== -->
## 16. Text Recurrent Neural Network (RNN) / Long Short-Term Memory (LSTM) over tokens

### Why we care

Text is sequential: the meaning of a word often depends on what came before it. An RNN reads text one token at a time while maintaining a **hidden state**, a compact memory of previous tokens. This helps with tasks such as predicting the next word, classifying a message, or detecting its sentiment.

Basic RNNs can gradually “forget” early information, especially in long texts. **LSTM** networks were designed to preserve important information for longer periods by learning what to remember, update, and discard.

### One-sentence intuition

An RNN reads a text like a person following a story, while an LSTM adds a carefully managed notebook that decides which details are worth keeping.

### How it works

First, a tokenizer converts text into tokens, such as words or subwords. Each token is represented numerically. The RNN processes the tokens in order:

1. It reads the current token.
2. It combines that token with its previous memory.
3. It produces a new memory and possibly a prediction.

An LSTM has three learned “gates”:

- A **forget gate** decides which old information to remove.
- An **input gate** decides which new information to store.
- An **output gate** decides which information to use immediately.

During training, the model compares its prediction with the correct answer and adjusts its parameters to make future predictions better.

### Everyday analogy

Imagine listening to a long conversation about an election. You remember the speaker’s main concern, such as security or employment, but forget unimportant words. An RNN tries to maintain this memory automatically; an LSTM has separate controls for erasing, writing, and rereading notes.

### On our Mexican elections

Suppose each row contains a Mexican voter’s written response, campaign message, or social-media post. The text could be represented as a sequence of tokens and passed through an LSTM to predict:

- The topic discussed, such as security, corruption, or healthcare.
- The sentiment toward a candidate.
- Whether a message supports, opposes, or is undecided about a political proposal.

The LSTM’s prediction would be based on word order and context. For example, it should distinguish “I support the proposal” from “I do not support the proposal.” It could also process a sequence of posts over time, although interpreting individual voters requires caution because the model may learn patterns associated with region, writing style, or demographic groups rather than political beliefs themselves.

### Key paper

Hochreiter and Schmidhuber (1997), *Long Short-Term Memory*. The paper introduced LSTM’s gated mechanism for learning dependencies across long time intervals.[1]

---

## 17. Byte-Pair Encoding (BPE) tokenizer and Out-of-Vocabulary (OOV)

### Why we care

A computer cannot directly understand words as humans do; it needs numerical tokens. A tokenizer decides how text is divided before a language model processes it.

If a tokenizer stores only complete words, it may encounter an unfamiliar word—an **out-of-vocabulary (OOV)** word. This is especially common with names, place names, misspellings, slang, indigenous-language terms, and newly created words. BPE reduces this problem by representing rare words as smaller, reusable pieces.

### One-sentence intuition

BPE learns common pieces of words so that an unfamiliar word can be assembled from parts instead of being replaced by an unknown symbol.

### How it works

BPE begins with small units, often individual characters or bytes. It then:

1. Counts which neighboring units occur together frequently.
2. Merges the most frequent pair into a larger unit.
3. Repeats this process many times.
4. Uses the resulting vocabulary to tokenize new text.

For example, after training, a tokenizer might represent:

- “candidate” as one token.
- “candidatura” as “candid” + “atura.”
- An unfamiliar surname as several smaller pieces.

An OOV problem occurs when a complete word is absent from a word-based vocabulary. With subword BPE, truly unknown words are much less common because many words can be decomposed into familiar pieces. Byte-level systems can usually represent any text, although the resulting sequence may be long.

### Everyday analogy

Imagine building words with LEGO blocks. A traditional word tokenizer keeps a separate complete LEGO model for every word. BPE keeps common blocks—such as “ción,” “mente,” or “pre”—and combines them to build unfamiliar words.

### On our Mexican elections

Mexican electoral data may contain:

- Candidate names and party names.
- Municipalities and states.
- Spanish accents and punctuation.
- Slang, abbreviations, hashtags, and spelling mistakes.
- Indigenous-language names or multilingual text.

A word-based tokenizer might treat “MORENA,” “morena,” and “Morena2024” as unrelated or mark some as unknown. BPE can split them into recognizable pieces, allowing a model to process rare candidate names and local place names more robustly.

However, token pieces are not automatically meaningful political concepts. A model should still be evaluated carefully, especially when comparing regions or groups with different naming conventions.

### Key paper

Sennrich, Haddow, and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units*. The paper applied BPE-style subword segmentation to reduce unknown-word problems in neural translation.

---

## 18. N-gram language models and perplexity

### Why we care

A language model assigns probabilities to sequences of words. For example, after “The candidate announced,” it may consider “a policy” more likely than “blue furniture.”

An **n-gram model** is an early and simple language model that predicts the next word using only the previous \(n-1\) words. It is useful because it provides an understandable baseline against which more complex models can be compared.

**Perplexity** measures how surprised the model is by a test text. Lower perplexity generally means the model predicted the text better.

### One-sentence intuition

An n-gram model guesses the next word by looking at a short recent window, while perplexity summarizes how often those guesses were wrong or uncertain.

### How it works

A bigram model uses one previous word:

\[
P(w_t \mid w_{t-1})
\]

A trigram model uses two:

\[
P(w_t \mid w_{t-2}, w_{t-1})
\]

The model estimates these probabilities by counting occurrences in training text. If “electoral participation” appears frequently, the model assigns a relatively high probability to “participation” after “electoral.”

A major difficulty is **data sparsity**: many word combinations never appear in the training data. Smoothing methods assign small nonzero probabilities to unseen combinations.

Perplexity is based on the model’s average probability for the correct next tokens. Informally:

- Low perplexity: the model was confident and usually correct.
- High perplexity: the text was surprising to the model.

Perplexity is meaningful mainly when comparing models on the same dataset, tokenization, and evaluation procedure.

### Everyday analogy

Suppose you hear someone say, “For dinner I would like…” You can guess “tacos” or “pizza” using the last few words. An n-gram model makes similar guesses, but it usually cannot remember much beyond its short window.

### On our Mexican elections

An n-gram model could be trained on campaign speeches, news articles, or voter comments from Mexico. It might learn that phrases such as:

- “seguridad pública”
- “participación ciudadana”
- “derechos humanos”

occur frequently.

You could then calculate perplexity separately for texts from different states, parties, or election years. A lower perplexity for one group might mean that its language resembles the training data more closely—not that the group is more coherent, truthful, or politically correct.

N-grams also provide a useful baseline. If an LSTM or another model does not substantially improve predictive performance over an n-gram model, the additional complexity may not be justified.

### Key paper

Kneser and Ney (1995), *Improved Backing-Off for M-Gram Language Modeling*. This work developed influential smoothing techniques for handling word sequences that are rare or absent in training data.

---

## 19. Word embeddings: Word2Vec / GloVe

### Why we care

Traditional text representations often treat words as unrelated labels. For example, “candidate,” “candidates,” and “politician” may receive completely separate identifiers.

A **word embedding** represents each word as a vector—a list of numbers—so that words used in similar contexts tend to have similar vectors. These vectors allow algorithms to calculate approximate semantic and linguistic relationships.

### One-sentence intuition

Words that appear in similar neighborhoods receive similar numerical coordinates.

### How it works

**Word2Vec** learns word vectors by solving a prediction task. In one common version, the model uses a word to predict nearby words. In another, it uses nearby words to predict the central word. During this process, words appearing in similar contexts acquire similar vectors.

**GloVe** learns vectors from global word co-occurrence statistics. It examines how often words appear together across the entire training collection and learns vectors that reflect those patterns.

After training, vector operations may reveal relationships such as:

\[
\text{vector}(\text{king})-\text{vector}(\text{man})+\text{vector}(\text{woman})
\]

being close to the vector for “queen.” Such relationships are approximate patterns, not dictionary definitions.

Embeddings inherit the biases of their training data. If political text repeatedly associates a group with negative language, the resulting vectors may reflect that association.

### Everyday analogy

Imagine placing words on a large map. Words with similar meanings or uses appear near one another, while unrelated words appear farther apart. The map is not designed by a dictionary; it is inferred from observing which words tend to travel together.

### On our Mexican elections

You could train embeddings on Mexican campaign speeches, news articles, or voter comments. Words such as “corrupción,” “sobornos,” and “transparencia” might form a neighborhood because they occur in related discussions. Embeddings could help with:

- Finding similar political terms.
- Grouping comments by topic.
- Representing text before classification.
- Comparing how political language changes between election years.

The interpretation must be contextual. “Cambio” could refer to political change, currency exchange, or small money depending on surrounding words. Also, an embedding trained on one election may encode that election’s parties, controversies, and media biases.

### Key paper

Mikolov, Chen, Corrado, and Dean (2013), *Efficient Estimation of Word Representations in Vector Space*, introduced the Word2Vec family of efficient predictive embedding methods. GloVe was later presented by Pennington, Socher, and Manning (2014) as a global co-occurrence-based alternative.[2]

---

## 20. t-SNE / UMAP for visualization

### Why we care

Word embeddings and document representations may contain dozens, hundreds, or thousands of numerical dimensions. Humans cannot easily inspect such vectors directly.

**t-SNE** and **UMAP** reduce high-dimensional data to two or three dimensions so that we can draw it. They are mainly visualization tools: they help us explore possible clusters and relationships, but they do not automatically prove that meaningful political groups exist.

### One-sentence intuition

These methods create a small map that tries to keep nearby high-dimensional points nearby, making hidden structure easier to inspect.

### How it works

Imagine each document as a point in a very large space. Documents with similar embeddings are close; dissimilar documents are far apart.

- **t-SNE** focuses strongly on preserving local neighborhoods. It tries to ensure that points close together in the original space remain close in the visualization.
- **UMAP** also emphasizes local neighborhoods but uses a mathematical model of the data’s underlying structure and often preserves more broader organization than t-SNE.

Both methods involve choices such as the number of neighbors, distance metric, random seed, and output dimensions. Different settings can produce different-looking maps. Apparent cluster size, distance between clusters, and empty space should therefore not be interpreted too literally.

### Everyday analogy

Suppose you have a huge map of every person’s interests, with thousands of dimensions—music, food, sports, books, and more. t-SNE or UMAP creates a small poster placing people with similar interests near one another so you can see groups, even though some details must be lost.

### On our Mexican elections

First, convert each electoral text—such as a voter comment, speech, or news article—into an embedding. Then apply t-SNE or UMAP to produce a two-dimensional plot.

You might observe regions containing comments about:

- Security and violence.
- Jobs and the economy.
- Healthcare.
- Corruption.
- Candidate-specific slogans.

You could color points by state, party mentioned, sentiment label, or election year and investigate whether those labels align with visible regions.

A visible cluster does not automatically represent a real ideological group. It might instead reflect writing length, repeated campaign slogans, language differences, hashtags, or duplicated posts. To test an interpretation, repeat the visualization with different settings and examine representative texts from each region.

### Key paper

van der Maaten and Hinton (2008), *Visualizing Data using t-SNE*, introduced t-SNE as a method for visualizing high-dimensional data. For UMAP, see McInnes, Healy, and Melville (2018), *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*.
<!-- ======== NLP 21-26 ======== -->
# 21. Latent Dirichlet Allocation (LDA) topic modeling

### Why we care

Political text often discusses several ideas at once: the economy, public safety, corruption, migration, or healthcare. **LDA** helps discover these recurring themes without requiring a researcher to label every document in advance. It represents each document as a mixture of topics rather than forcing it into only one category.[3]

### One-sentence intuition

**LDA assumes that documents are mixtures of hidden topics, and topics are mixtures of words.**

### How it works

Imagine that the model must explain a collection of speeches:

1. It guesses several topics.
2. Each topic receives high probabilities for certain words—for example, *hospital*, *medicine*, and *doctors* may form a healthcare topic.
3. Each document receives a mixture of topics—for example, one speech might be 50% security, 30% economy, and 20% healthcare.
4. The model repeatedly adjusts these probabilities so that the topics explain the words appearing together across documents.

The topics are called “latent” because they are not directly observed. The researcher interprets them by examining their most probable words. LDA does not automatically know that a topic is “corruption”; that label is assigned by a human after inspecting the words.

### Everyday analogy

Suppose you find several mixed bowls of colored candies. You do not know the original recipes, but by seeing which colors frequently occur together, you infer hidden mixtures such as “mostly red and yellow” or “mostly blue and green.” LDA performs a similar decomposition with words.

### On our Mexican elections

Each document could be a candidate speech, party platform, debate answer, campaign post, or news statement. LDA might identify topics such as:

- public security and organized crime;
- jobs, inflation, and economic growth;
- corruption and government accountability;
- social programs and inequality;
- migration and relations with the United States.

For each document, we could estimate the proportion devoted to each topic and compare those proportions across candidates, parties, states, or election years. These are statistical themes, not proof of a candidate’s intentions; interpretation and validation remain necessary.

### Key paper

**Blei, Ng, and Jordan (2003)**, “Latent Dirichlet Allocation.”[3]

# 22. Structural Topic Model (STM)

### Why we care

LDA discovers topics but does not naturally use information about the documents, such as the candidate, party, election year, or state. **STM** extends topic modeling by incorporating such metadata, allowing us to study how political language changes across groups and contexts.[1][5]

### One-sentence intuition

**STM asks not only “What topics appear?” but also “Who discusses them, when, and in what way?”**

### How it works

STM still represents each document as a mixture of topics, but it adds document-level information called **covariates**.

For example:

- topic prevalence can vary by candidate, party, or year;
- the words used to express a topic can also vary by party or election period;
- topics may be correlated—for example, security and corruption may frequently appear together.

A model might find that all parties discuss security, but one party emphasizes *police* and *punishment*, while another emphasizes *prevention* and *social investment*. Thus, STM can study both **how much** attention a topic receives and **how the topic is expressed**.[5]

### Everyday analogy

Imagine comparing restaurant menus. LDA identifies common dishes, while STM also records the restaurant’s location, price range, and type of cuisine. It can then ask whether expensive restaurants serve more seafood and whether different restaurants describe similar dishes differently.

### On our Mexican elections

We could include:

- candidate or party;
- election year;
- state or region;
- document type, such as debate speech or party platform;
- campaign period.

STM could estimate whether security became more prominent over time, whether parties differed in their attention to social programs, or whether candidates used different vocabulary when discussing the same topic. For instance, the model might compare the language of “security” in northern states with that in southern states.

These are associations, not automatically causal effects. A candidate may differ from another for many reasons besides party identity.

### Key paper

**Roberts et al. (2014)**, “Structural Topic Models for Open-Ended Survey Responses.”[1][5]

# 23. Text classification / stance detection

### Why we care

Sometimes we already know the categories we care about. We may want to determine whether a statement supports, opposes, or is neutral toward a proposal. This is the task of **text classification**; when the categories describe a position toward a specific target, it is called **stance detection**.

### One-sentence intuition

**A classifier learns from labeled examples to predict the category or position of a new text.**

### How it works

First, humans create training examples. For instance:

- “We support expanding public healthcare” → *support*;
- “This reform would harm families” → *oppose*;
- “The bill was introduced yesterday” → *neutral*.

The computer converts words and their context into numerical representations. It then learns patterns connecting those representations to labels. After training, it predicts labels for unseen statements.

A good stance system must identify the **target**. A sentence may support one issue but oppose another. It must also handle negation, sarcasm, indirect language, and quotations. Performance is measured on held-out examples using metrics such as accuracy, precision, recall, and F1 score.

### Everyday analogy

A student learns to sort mail into folders by studying many examples labeled “school,” “bills,” and “family.” Afterward, the student places a new letter into the most appropriate folder. Stance detection is similar, except the folders might be “supports,” “opposes,” and “neutral toward a policy.”

### On our Mexican elections

We could classify statements about:

- a candidate;
- a party;
- a policy such as energy reform or social programs;
- an institution such as the electoral authority.

Possible labels might be:

- supports the target;
- opposes the target;
- neutral or merely mentions it.

For example, the system could analyze campaign posts to estimate how often each party supports or attacks a proposal. Human-labeled examples should include regional language, slang, sarcasm, and quoted statements. Otherwise, the model may learn superficial cues—such as associating a particular candidate’s name with a stance—rather than understanding the statement.

### Key paper

A commonly cited foundational reference for stance detection is **Mohammad et al. (2016)**, “SemEval-2016 Task 6: Detecting Stance in Tweets.”

# 24. Named Entity Recognition (NER) and BIO tags

### Why we care

Political texts contain important names and organizations: candidates, parties, states, municipalities, institutions, and countries. **Named Entity Recognition (NER)** identifies these mentions and assigns them types.

### One-sentence intuition

**NER highlights who or what is being mentioned, while BIO tags show where each entity begins and ends.**

### How it works

NER labels individual tokens, or pieces of text. The common **BIO** system uses:

- **B** = beginning of an entity;
- **I** = inside or continuation of an entity;
- **O** = outside any entity.

Example:

| Token | Tag |
|---|---|
| Claudia | B-PER |
| Sheinbaum | I-PER |
| visitó | O |
| Ciudad | B-LOC |
| de | I-LOC |
| México | I-LOC |

Here, `PER` means person and `LOC` means location. Other categories could include organization, party, institution, or policy.

BIO tags are useful because they distinguish separate entities and multiword names. For example, “Partido Acción Nacional” should be treated as one organization rather than several unrelated words.

### Everyday analogy

Imagine highlighting a newspaper with different colored markers: blue for people, green for places, and red for organizations. BIO tags provide a precise machine-readable version of those highlights, including the exact beginning and ending of each highlighted phrase.

### On our Mexican elections

NER could identify:

- presidential and local candidates;
- political parties and coalitions;
- states, cities, and municipalities;
- electoral institutions;
- countries and international organizations.

This would allow us to count mentions, build networks of co-occurring actors, and study which candidates are associated with which places or institutions. Because Mexican names and locations can be ambiguous, the system should be evaluated on manually annotated Mexican electoral text.

### Key paper

**Ramshaw and Marcus (1995)**, “Text Chunking Using Transformation-Based Learning,” helped establish the B-I-O style of sequence labeling for identifying structured spans in text. The BIO schema remains widely used for marking entity beginnings, continuations, and nonentity tokens.[7]

# 25. Conditional Random Field (CRF)

### Why we care

NER and similar tasks require assigning a label to every token. A token’s correct label often depends on neighboring labels. For example, an `I-PER` tag should usually follow `B-PER` or another `I-PER`, not `O`. A **Conditional Random Field (CRF)** models these relationships explicitly.

### One-sentence intuition

**A CRF chooses the most plausible sequence of labels for the entire sentence, rather than labeling each word independently.**

### How it works

Suppose the sentence is:

> “Partido Acción Nacional ganó.”

A CRF considers:

- the words themselves;
- features of each word, such as capitalization or spelling;
- the label of neighboring words;
- whether a sequence such as `B-ORG I-ORG` is plausible.

It assigns a score to possible complete label sequences and selects the sequence with the highest probability. During training, it learns which word patterns and label transitions are common.

This is especially helpful because language has structure. A model might learn that `B-ORG` is often followed by `I-ORG`, while `I-ORG` should rarely begin a sentence.

### Everyday analogy

A person completing a form does not decide each box independently. If one box says “street name,” the next boxes are more likely to contain the rest of the address. A CRF makes similar decisions while considering the whole sequence.

### On our Mexican elections

A CRF could label each token in campaign texts as:

- person;
- organization or political party;
- location;
- institution;
- non-entity.

It could learn that “Partido” followed by several capitalized words is likely an organization, or that a candidate’s full name should receive one continuous entity label. It could also help identify multiword locations and party coalitions.

CRFs are often placed after a neural language model, such as BERT, to improve sequence consistency. However, performance depends on high-quality annotations and representative training data.

### Key paper

**Lafferty, McCallum, and Pereira (2001)**, “Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.”[14]

# 26. BETO / BERT / Masked Language Modeling (MLM)

### Why we care

Older methods often treat words as isolated symbols or use limited context. **BERT** creates context-sensitive representations: the meaning of a word can change depending on the surrounding sentence. **BETO** is a BERT-style model pretrained specifically for Spanish, making it useful for Spanish-language Mexican electoral data.

### One-sentence intuition

**BERT learns language by looking at both sides of a sentence and predicting words that have been hidden.**

### How it works

BERT uses a neural-network architecture called a **Transformer**, which allows each word to pay attention to other relevant words in the sentence.

During **masked language modeling**:

1. Some words are hidden, or “masked.”
2. The model reads the surrounding words.
3. It predicts the missing words.
4. Repeating this over enormous text collections teaches the model grammar, vocabulary, and many contextual associations.

For example:

> “El candidato propuso aumentar el ___ mínimo.”

The model may predict *salario* because the surrounding words provide clues.

After pretraining, BERT can be **fine-tuned** for a particular task using labeled data, such as stance detection, sentiment classification, or NER. BETO applies this approach to Spanish and is therefore better suited than an English-only model to Spanish morphology, vocabulary, and syntax.

### Everyday analogy

Imagine a student repeatedly receiving sentences with a few words erased. By guessing the missing words and seeing the correct answers, the student gradually learns how language works. Later, the student can use that knowledge to classify statements or identify names.

### On our Mexican elections

BETO could be fine-tuned to:

- detect support or opposition toward electoral issues;
- classify campaign messages by topic;
- identify candidates, parties, and locations;
- analyze speeches, debates, social-media posts, and news;
- compare how candidates frame the same policy.

Because Mexican electoral language includes slang, regional expressions, code-switching, misspellings, and political sarcasm, the model should be tested on a representative sample of the actual dataset. A prediction is not automatically an explanation: researchers should inspect errors and, where possible, provide evidence for why the model assigned a label.

### Key paper

For BERT, the foundational citation is **Devlin et al. (2019)**, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.”[8][12] For the Spanish model BETO, a commonly cited reference is **Cañete et al. (2020)**, “Spanish Pre-Trained BERT Model and Evaluation Data.”
