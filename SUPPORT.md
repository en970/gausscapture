# Support and maintenance status

## Bus factor: 1

GaussCapture is maintained by one person, [Enes Öz](https://github.com/en970), as an independent open-source research project. There is no company behind it and no service-level commitment.

This is stated up front deliberately. Projects in this space have lost credibility by leaving users to discover a maintenance gap on their own; publishing the status is more useful than implying a team that does not exist.

## What that means in practice

| | |
|---|---|
| Bug reports | Read; triaged when possible. No response-time guarantee. |
| Security issues | Prioritised. Email `enesozile@gmail.com` rather than opening a public issue. |
| Feature requests | Weighed against [the roadmap](docs/ROADMAP.md). Requests outside its scope will be closed with a reason, not left open indefinitely. |
| Pull requests | Welcome. Small, focused ones get merged fastest. |
| Breaking changes | Expected while the version stays `0.y.z`. The `.capturepack` format in particular will change. |

## Getting help

1. Check [`docs/RESEARCH.md §2`](docs/RESEARCH.md) — several known failures are documented there rather than being bugs to report.
2. Search existing issues.
3. Open a new issue including: your OS and hardware, the output of `ffmpeg -version` and `colmap -h`, the failing step, and the job log from the UI.

## Getting help faster

Capture failures are the most valuable reports. If a reconstruction came out poorly, attach the quality report JSON and describe how you filmed it — lighting, walking speed, whether exposure was locked, how many orbits. That is exactly the data this project is trying to turn into a predictor.

## If maintenance stops

If this project becomes unmaintained, this file will say so plainly and point to alternatives.
