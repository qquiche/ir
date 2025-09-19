package ir.vsr;

import java.io.*;

/**
 * An HTML document that does NOT remove stopwords or non-letters.
 * Strips HTML tags like HTMLFileDocument but keeps all tokens.
 * Used only for positional indexing (so distances reflect actual spacing).
 */
public class RawHTMLFileDocument extends HTMLFileDocument {

  public RawHTMLFileDocument(File file) {
    super(file, false); // stem=false, but we override prepareNextToken
  }

  @Override
  protected void prepareNextToken() {
    // Just take the next candidate, lowercase, and return it — no stopword removal
    nextToken = getNextCandidateToken();
    if (nextToken == null) return;
    nextToken = nextToken.toLowerCase();
    // Optionally still filter out non-letters
    if (!allLetters(nextToken)) nextToken = null;
  }
}
