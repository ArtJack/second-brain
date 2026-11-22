import { describe, expect, it } from "vitest";

import { isUnsupported } from "./grounding";

describe("isUnsupported", () => {
  it("warns when evidence was retrieved and the answer cited none of it", () => {
    expect(isUnsupported({ sources_retrieved: 5, citations_present: false, refused: false, unsupported: true })).toBe(
      true,
    );
  });

  it("stays silent on a declared refusal, which is an honest answer", () => {
    expect(isUnsupported({ sources_retrieved: 5, citations_present: false, refused: true, unsupported: false })).toBe(
      false,
    );
  });

  it("stays silent when there was nothing to cite", () => {
    expect(isUnsupported({ sources_retrieved: 0, citations_present: false, refused: false, unsupported: false })).toBe(
      false,
    );
  });

  it("stays silent when the answer cited its sources", () => {
    expect(isUnsupported({ sources_retrieved: 5, citations_present: true, refused: false, unsupported: false })).toBe(
      false,
    );
  });

  it("says nothing rather than alarming when the server sends no grounding", () => {
    expect(isUnsupported(undefined)).toBe(false);
    expect(isUnsupported(null)).toBe(false);
  });

  it("falls back to the parts when an older server sends no verdict", () => {
    expect(isUnsupported({ sources_retrieved: 5, citations_present: false })).toBe(true);
    expect(isUnsupported({ sources_retrieved: 5, citations_present: false, refused: true })).toBe(false);
  });
});
