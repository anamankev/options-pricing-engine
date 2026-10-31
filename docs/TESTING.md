# Testing it yourself

This guide shows how to check, on your own computer, that everything this project claims is true, and how to try it with your own inputs. No coding is needed: you type commands and read the results.

Where to type the commands: in VS Code, open the project folder (File > Open Folder... > `options-pricing-engine`), then open a terminal with Terminal > New Terminal. Paste each command and press Enter.

## 1. One-time setup

If you have not set the project up yet, run these once (macOS or Linux):

```bash
git clone https://github.com/anamankev/options-pricing-engine.git
cd options-pricing-engine
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,oracle,app]"
```

What each part installs:
- `dev`: the test tools.
- `oracle`: QuantLib and Let's Be Rational, the independent libraries everything is compared with.
- `app`: the playground in section 4.

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\activate` instead of the third and fourth lines.

Every time you open a new terminal, activate the environment first:

```bash
source .venv/bin/activate
```

You will see `(.venv)` at the start of the prompt. That means it worked.

## 2. Check everything: `python -m verify`

```bash
python -m verify
```

It takes about 20 seconds and prints one block per check. For example:

```
[PASS] Black-Scholes prices vs 50-digit mpmath (288 cases)
       worst error 0.76 eps x scale (README: 0.76)   (limit: <= 64)   [0.0 s]
...
Summary: 9 passed, 0 failed, 0 skipped
Everything passed.
```

How to read it:
- **PASS** means the measured number is within the limit.
- **FAIL** means it is not, and the line says by how much.
- **SKIP** means the check could not run, and says why. Usually QuantLib is not installed.
- **"eps"** is the smallest relative difference two computer numbers can have, about 2.2e-16. An error of "0.76 eps" means the answer is as exact as a computer number allows.

The nine checks:
1. **Test suite:** every automated test in the repository.
2. **README numbers:** every number in the README's Status and Results sections is compared with the result file it comes from. The list lives in `docs/claims.toml`.
3. **Black-Scholes prices** against the same formula computed with 50 correct digits.
4. **The six Greeks** against finite differences, which estimate the same derivative numerically.
5. **Implied volatility:** price an option, back the volatility out of the price, and check it comes back.
6. **American options** against the 18 published values of Ju and Zhong (1999).
7. **Prices and Greeks against QuantLib**, the industry's standard open-source library.
8. **Binomial trees against QuantLib's trees.**
9. **Monte Carlo:** 95% confidence intervals contain the exact price about 95% of the time.

To save the report as a file you can share:

```bash
python -m verify --report verification.md
```

## 3. Rerun every experiment: `python -m verify --full`

```bash
python -m verify --full
```

This takes about 6 minutes. On top of section 2, it reruns all nine experiments that produced the README's results.

Each result file records the exact version of the code (the git commit) that produced it. `--full` checks out that version in a temporary folder, reruns the experiment there, and compares every number with the committed file, so a PASS means the published numbers are exactly what the code produces. Your own copy of the project is not changed.

Timings are not compared, because they depend on the computer and on what else it is doing. The report says what was left out for each experiment.

## 4. Try your own inputs: the playground

```bash
streamlit run app/playground.py
```

Your browser opens at `http://localhost:8501`. The page runs only on your computer; nothing is sent anywhere. Stop it with Ctrl+C in the terminal.

On the left, type in an option. The defaults are Hull's textbook example (Example 15.6): a call, S = 42, K = 40, 180 days on a 360-day year (T = 0.5), r = 0.10, sigma = 0.20. The four tabs:

- **Price and Greeks.** This library's price and Greeks beside QuantLib's, the difference, and PASS or FAIL. For the defaults the price is 4.7594 in both columns, and all seven rows say PASS.
- **Trees (American).** How the four binomial methods converge as the number of steps N grows, against an accurate reference line.
- **Monte Carlo.** A simulated price with its 95% interval, and whether the interval contains the exact price.
- **Implied volatility.** Type a price and every solver returns the volatility, with the number of iterations and why it stopped.

## 5. Try to break it

Each exercise below gives you something to do and the result you should see. If you see something else, that is a finding worth reporting.

1. **QuantLib fails in a far tail, this library does not.** In the playground, set Put, S = 130, K = 100, 7 days, 365-day year, r = -0.01, q = -0.01, sigma = 0.10.
   - The price tab shows this library at about 2.0e-81 and QuantLib at exactly 0.
   - The true value is 2.0e-81 (the README's validation section and `docs/notes/phase_1_quantlib_reconciliation.md` explain this).
   - The row still says PASS, because the two differ by far less than a cent. Agreement is judged on absolute size when values are this small.
2. **An American call without dividends equals the European call.** With the default inputs (q = 0), open the Trees tab.
   - Every method converges to the reference, 4.7594: the same as the European price.
   - With no dividend it never pays to exercise a call early.
3. **Wrong inputs are refused, by name.** In the terminal, run:
   ```bash
   python -c "from pricing import black_scholes as bs; from pricing.contracts import OptionType; bs.price(OptionType.CALL, S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=-0.2)"
   ```
   The last line reads `PricingInputError: sigma must be >= 0; got -0.2`. The code never silently "fixes" a bad input.
4. **CRR zigzags.** In the Trees tab, the CRR line jumps up and down between odd and even N. Leisen-Reimer and BBSR are smooth. This is Study B's first finding.
5. **One Monte Carlo miss proves nothing.**
   - With the defaults, seed 1's interval misses the exact price, as about 1 seed in 20 should.
   - Click **Run 200 seeds**: the share that cover it comes out near 0.95 (0.945 on the defaults).
6. **An impossible price is rejected.**
   - In the Implied volatility tab, type 1.0 as the price. It is below the call's intrinsic value (S minus the discounted strike is about 3.95), so no volatility can produce it.
   - Every solver says "rejected" and gives the reason, instead of returning a made-up number.
7. **A mistyped README number is caught.**
   - Open `README.md`, find "The worst error is 0.76", change 0.76 to 0.67 and save.
   - Run `python -m verify`. Two checks FAIL: the test suite, and the README numbers check, which says:
     ```
     validation.mpmath_worst: phrase not found in README.md
     number '0.67' has no source, in: - **Exact values:** ... The worst error is 0.67 eps ...
     ```
     The sentence the ledger expects is gone, and 0.67 has no source.
   - Undo the change with `git checkout README.md`.

## 6. Rerun a single study

Each result in the README comes from one command, listed in the README's "Reproducing the results" table. For example, Study B:

```bash
python -m experiments.study_b_trees --config experiments/configs/study_b_trees.toml
```

It rewrites the study's tables and figures under `experiments/results/`. `git diff --stat` then shows which files changed. Only timing columns should differ, because they depend on your computer. To put the committed files back:

```bash
git checkout experiments/results
```

Each study's write-up is in `docs/notes/`: the question, the hypotheses (committed before the study ran), the design, the results, and which hypotheses failed.
