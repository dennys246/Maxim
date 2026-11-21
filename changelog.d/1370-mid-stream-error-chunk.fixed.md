- **A self-hosted model's stream that ends in an error no longer comes back as a complete answer** (#1370): an
  error chunk in the middle of a stream (`data: {"error": …}`) used to be ignored, and the text streamed before it
  was returned as a successful completion. It now fails the call, so the router moves to the next provider, and the
  partial text is discarded. A stream chunk that is JSON but not an object now fails the same typed way, instead of
  escaping as a raw `AttributeError`.
