import React, { useState, useEffect, useRef } from 'react';
import { Send, MapPin, Activity, Cloud, Thermometer, Droplets, Navigation, Hospital, Car } from 'lucide-react';

const App = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "👋 Hello! I'm your GIS Assistant. I can help you analyze spatial data, create maps, and visualize geographic information. Try asking me about hospitals in Chennai, rainfall patterns, or road networks in any city!",
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [mapData, setMapData] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [status, setStatus] = useState('Ready');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sampleQueries = [
    { text: "Show hospitals in Chennai with 15km radius", icon: <Hospital className="w-4 h-4" /> },
    { text: "Rainfall patterns in Mumbai", icon: <Cloud className="w-4 h-4" /> },
    { text: "Road network in Bangalore", icon: <Car className="w-4 h-4" /> },
    { text: "Temperature trends in Delhi", icon: <Thermometer className="w-4 h-4" /> }
  ];

  const sendMessage = async (text = inputText) => {
    if (!text.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: text.trim(),
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);
    setStatus('Processing query...');

    try {
      const response = await simulateBackendCall(text);

      setTimeout(() => {
        const botMessage = {
          id: Date.now() + 1,
          text: response.message,
          isUser: false,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botMessage]);
        setMapData(response.mapData);
        setGraphData(response.graphData);
        setIsTyping(false);
        setStatus('Analysis complete');
      }, 1500);
    } catch (error) {
      console.error('Error calling backend:', error);
      setIsTyping(false);
      setStatus('Error occurred');
    }
  };

  const simulateBackendCall = async (query) => {
    const response = await fetch('http://localhost:5000/api/spatial-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    return await response.json();
  };

  return (
    <div className="h-screen w-full flex items-center justify-center text-center p-10">
      <div>
        <h1 className="text-2xl font-bold mb-6">🌍 GIS Assistant</h1>
        <div className="mb-4">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={(e) => { if (e.key === 'Enter') sendMessage(); }}
            className="border px-4 py-2 rounded-lg w-96"
            placeholder="Ask me a spatial question..."
          />
          <button
            onClick={() => sendMessage()}
            className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-lg"
          >
            <Send className="inline w-4 h-4" />
          </button>
        </div>
        <div className="border rounded-lg p-4 h-64 overflow-y-scroll bg-gray-100">
          {messages.map((msg) => (
            <div key={msg.id} className={`mb-2 text-left ${msg.isUser ? 'text-blue-600' : 'text-black'}`}>
              <strong>{msg.isUser ? 'You' : 'Bot'}:</strong> {msg.text}
            </div>
          ))}
          {isTyping && <div className="text-gray-400">Bot is typing...</div>}
          <div ref={messagesEndRef} />
        </div>
        <p className="mt-4 text-sm text-gray-500">Status: {status}</p>
      </div>
    </div>
  );
};
console.log("🌍 GISChatbot Loaded");

export default App;
