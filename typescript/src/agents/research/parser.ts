/** Parse the strict-format final response into structured fields. */

export interface ParsedResearch {
  readonly answer: string;
  readonly explanation: string;
  readonly confidence: number | null;
  readonly citationUrls: ReadonlyArray<string>;
}

const FIELD_PATTERNS: Record<string, RegExp> = {
  explanation: /^\s*explanation\s*:\s*([\s\S]+?)(?=^\s*(?:exact\s+answer|confidence|citations)\s*:|\s*$)/im,
  answer: /^\s*exact\s+answer\s*:\s*([\s\S]+?)(?=^\s*(?:explanation|confidence|citations)\s*:|\s*$)/im,
  confidence: /^\s*confidence\s*:\s*([\s\S]+?)(?=^\s*(?:explanation|exact\s+answer|citations)\s*:|\s*$)/im,
  citations: /^\s*citations\s*:\s*([\s\S]+?)\s*$/im,
};

function grab(text: string, name: string): string {
  const m = FIELD_PATTERNS[name].exec(text);
  return m ? m[1].trim() : '';
}

export function parseResearchOutput(text: string): ParsedResearch {
  if (!text) {
    return { answer: '', explanation: '', confidence: null, citationUrls: [] };
  }

  const answer = grab(text, 'answer');
  const explanation = grab(text, 'explanation');
  const confidenceRaw = grab(text, 'confidence');
  const citationsRaw = grab(text, 'citations');

  let confidence: number | null = null;
  if (confidenceRaw) {
    const m = /(\d+(?:\.\d+)?)/.exec(confidenceRaw);
    if (m) {
      const v = parseFloat(m[1]);
      confidence = v > 1 ? v / 100 : v;
    }
  }

  const citationUrls: string[] = [];
  if (citationsRaw) {
    const urlRe = /https?:\/\/[^\s,;<>"')]+/g;
    let m: RegExpExecArray | null;
    while ((m = urlRe.exec(citationsRaw)) !== null) {
      if (!citationUrls.includes(m[0])) citationUrls.push(m[0]);
    }
  }

  return { answer, explanation, confidence, citationUrls };
}
