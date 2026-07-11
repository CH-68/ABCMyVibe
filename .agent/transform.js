const { OpenAI } = require("openai"); 
const fs = require('fs');

# const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function transformFile(filePath) {
  const sourceCode = fs.readFileSync(filePath, 'utf8');
  
  const response = await openai.chat.completions.create({
    model: "nomic-embed-text", // or a fast local model like "llama3" via Ollama
    messages: [
      { role: "system", content: "You are an expert refactoring agent. Output ONLY valid raw code. No markdown wrapping, no explanations." },
      { role: "user", content: `Refactor this code to use TypeScript and add error handling:\n\n${sourceCode}` }
    ],
  });

  const upgradedCode = response.choices[0].message.content;
  fs.writeFileSync(filePath, upgradedCode, 'utf8');
  console.log(`Successfully transformed: ${filePath}`);
}
