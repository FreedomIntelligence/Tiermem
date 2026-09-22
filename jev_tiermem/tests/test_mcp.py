import importlib.util
import json
import sys
import tempfile
import unittest

from jev_tiermem import JevTierMem
from jev_tiermem.demo import DemoModel, DemoRouter


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional mcp extra is not installed")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_handshake_and_persistent_tools(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory() as directory:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "jev_tiermem.mcp_server", "--store", directory, "--session", "mcp-test"],
            )
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    initialized = await client.initialize()
                    self.assertEqual(initialized.serverInfo.name, "jev_tiermem")
                    tools = (await client.list_tools()).tools
                    self.assertEqual(len(tools), 8)
                    self.assertTrue(next(t for t in tools if t.name == "jev_memory_retrieve")
                                    .annotations.readOnlyHint)
                    observed = await client.call_tool("jev_memory_observe", {"text": "callback timeout 60"})
                    self.assertFalse(observed.isError)
                    ids = json.loads(observed.content[0].text)["raw_ids"]
                    summary = await client.call_tool("jev_memory_add_summary", {"text": "timeout 60", "raw_ids": ids})
                    self.assertFalse(summary.isError)
                    self.assertEqual(json.loads(summary.content[0].text)["status"], "written")
                    found = await client.call_tool("jev_memory_search", {"query": "callback"})
                    self.assertEqual(json.loads(found.content[0].text)["raw"][0]["id"], ids[0])
                    read = await client.call_tool("jev_memory_get", {"ids": ids})
                    self.assertEqual(json.loads(read.content[0].text)["raw"][0]["text"], "callback timeout 60")
                    bad = await client.call_tool("jev_memory_search", {"query": "callback", "limit": 0})
                    self.assertTrue(bad.isError)
            with JevTierMem(directory, "mcp-test") as reopened:
                self.assertEqual(reopened.store.notes()[0]["text"], "timeout 60")

    async def test_agent_retrieve_and_promote_use_same_pipeline(self):
        from jev_tiermem.mcp_server import create_server

        with tempfile.TemporaryDirectory() as directory:
            memory = JevTierMem(directory, "test", model=DemoModel(), router=DemoRouter())
            ids = memory.observe("Mira prefers email. Departure is 17:30.")
            memory.add_summary("Mira prefers email.", ids)
            server = create_server(memory)
            result = await server.call_tool("jev_memory_retrieve", {"query": "What time does Mira leave?"})
            payload = json.loads(result[0].text)
            self.assertEqual(payload["route"], "R")
            self.assertEqual(payload["evidence"][0]["id"], ids[0])
            promoted = await server.call_tool("jev_memory_promote", {
                "text": "Departure is 17:30.", "quotes": [{"raw_id": ids[0], "quote": "Departure is 17:30."}],
            })
            self.assertEqual(json.loads(promoted[0].text)["status"], "written")


if __name__ == "__main__":
    unittest.main()
