"use client";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

type Message = { role: "user" | "assistant"; content: string };

const SUGGESTIONS = [
  "What is the total revenue?",
  "Which customer segment generates most revenue?",
  "What is the only true Superstar product?",
  "Compare November 2010 to November 2011 revenue.",
];

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const send = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const assistantPlaceholder: Message = { role: "assistant", content: "" };
      setMessages((prev) => [...prev, assistantPlaceholder]);

      // 1. Build the full conversation history (including the new message)
      const conversationHistory: Message[] = [...messages, userMessage].map(
        (m): Message => ({ role: m.role, content: m.content })
      );

      // 2. Send the full history to the Azure backend
      const response = await fetch("https://retail-ai-api-pratik-g9fje5ayeyc8cpgm.koreacentral-01.azurewebsites.net/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: conversationHistory }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        aiResponse += decoder.decode(value, { stream: true });
        
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = { role: "assistant", content: aiResponse };
          return next;
        });
      }
    } catch {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          content: "Couldn't reach the agent. The API may be cold-starting — try again in a few seconds.",
        };
        return next;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <div className="relative flex flex-col h-screen bg-[#080808] text-slate-100 overflow-hidden">
      {/* Ambient Background Glows */}
      <div className="absolute top-0 left-1/4 w-[600px] h-[600px] rounded-full blur-[120px] bg-cyan-600/10 pointer-events-none animate-pulse" />
      <div className="absolute bottom-0 right-1/4 w-[600px] h-[600px] rounded-full blur-[120px] bg-blue-800/10 pointer-events-none animate-pulse" />

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between p-4 border-b border-white/5 backdrop-blur-xl bg-black/40">
        <h1 className="text-xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
          Retail Intelligence
        </h1>
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
          </span>
          <span className="text-xs text-slate-400 font-mono uppercase tracking-wider">Agent Online</span>
        </div>
      </div>

      {/* Chat Area */}
      <div ref={scrollRef} className="relative z-10 flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }} 
              animate={{ opacity: 1, scale: 1 }} 
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="flex flex-col items-center justify-center h-full pt-20"
            >
              <div className="w-24 h-24 rounded-full bg-cyan-500/10 flex items-center justify-center border border-cyan-500/30 mb-6 shadow-[0_0_40px_rgba(34,211,238,0.3)]">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h2 className="text-3xl font-semibold text-slate-100 mb-2 tracking-tight">Retail Revenue Intelligence</h2>
              <p className="text-slate-500 mb-10 text-center max-w-md">Your multi-agent analytics copilot. Ask anything about sales, products, or customers.</p>
              
              <div className="grid grid-cols-2 gap-3 w-full max-w-lg">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    className="text-left text-sm bg-slate-900/50 hover:bg-slate-800/80 border border-slate-700/50 rounded-xl px-4 py-3 transition-all hover:border-cyan-500/50 text-slate-300 hover:text-white backdrop-blur-md hover:scale-105 transform"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </motion.div>
          ) : (
            <AnimatePresence initial={false}>
              {messages.map((m, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div className={`max-w-[85%] rounded-2xl px-5 py-3 shadow-lg backdrop-blur-md ${
                    m.role === "user"
                      ? "bg-gradient-to-br from-cyan-600/90 to-blue-700/90 text-white rounded-br-none shadow-cyan-500/20"
                      : "bg-slate-900/80 border border-cyan-500/20 text-slate-100 rounded-bl-none shadow-black/50 shadow-[0_0_20px_rgba(34,211,238,0.05)]"
                  }`}>
                    <div className="text-xs font-bold mb-1 opacity-60 uppercase tracking-wider">
                      {m.role === "user" ? "You" : "AI Analyst"}
                    </div>
                    {m.role === "assistant" ? (
                      <div className="text-sm leading-relaxed font-light prose prose-invert prose-sm max-w-none prose-headings:my-2 prose-p:my-1 prose-ul:my-1 prose-pre:bg-[#1a1a1a] prose-pre:p-3 prose-pre:rounded-lg">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            code({ node, className, children, ...props }) {
                              const match = /language-(\w+)/.exec(className || '');
                              const { ref: _ref, ...syntaxProps } = props as any;

                              if (match) {
                                return (
                                  <SyntaxHighlighter
                                    style={oneDark as any}
                                    language={match[1]}
                                    PreTag="div"
                                    {...syntaxProps}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                );
                              }

                              return (
                                <code className={className} {...props}>
                                  {children}
                                </code>
                              );
                            }
                          }}
                        >
                          {m.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</div>
                    )}
                    {isLoading && index === messages.length - 1 && m.role === "assistant" && m.content === "" && (
                      <span className="inline-flex gap-1 mt-2">
                        <span className="h-2 w-2 bg-cyan-400 rounded-full animate-bounce delay-100"></span>
                        <span className="h-2 w-2 bg-cyan-400 rounded-full animate-bounce delay-300"></span>
                        <span className="h-2 w-2 bg-cyan-400 rounded-full animate-bounce delay-500"></span>
                      </span>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className="relative z-10 p-4 border-t border-white/5 backdrop-blur-xl bg-black/40">
        <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="max-w-3xl mx-auto flex items-center gap-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask the warehouse anything..."
            className="flex-1 bg-slate-900/80 border border-slate-700 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 rounded-xl px-4 py-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition-all font-light"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold px-6 py-3.5 rounded-xl transition-all duration-300 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/40 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed text-sm flex items-center gap-2"
          >
            {isLoading ? "Thinking..." : "Send"}
            {!isLoading && (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}