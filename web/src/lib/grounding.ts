import type { Grounding } from "../types/brain";

/** Whether the reader should be told this answer is not grounded in the sources.
 *
 * True when evidence was retrieved, the answer cited none of it, and it did not
 * declare that the notes do not cover the question. A refusal is an honest answer
 * and never carries the warning: a badge that fires on every "not in your notes"
 * is a badge readers learn to ignore, which is how the signal would be lost.
 */
export function isUnsupported(grounding?: Grounding | null): boolean {
  if (!grounding) return false;
  if (typeof grounding.unsupported === "boolean") return grounding.unsupported;
  // An older server sends the parts without the verdict.
  return (
    Boolean(grounding.sources_retrieved) &&
    grounding.citations_present === false &&
    grounding.refused !== true
  );
}
