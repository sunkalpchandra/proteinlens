import "@testing-library/jest-dom/vitest";

// jsdom lacks ResizeObserver / canvas; components under test that touch them
// get inert stand-ins.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as Record<string, unknown>).ResizeObserver ??= ResizeObserverStub;
