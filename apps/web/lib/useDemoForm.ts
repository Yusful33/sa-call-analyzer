import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";

/**
 * Zod schema for the Demo Builder form.
 * Validates required fields and provides defaults.
 */
export const demoFormSchema = z.object({
  accountName: z.string().min(1, "Account name is required"),
  additionalContext: z.string().optional(),
  industryOrUseCase: z.string().min(1, "Industry/use case is required"),
  outputDir: z.string().optional(),
  skillFramework: z.enum([
    "openai",
    "anthropic",
    "bedrock",
    "vertex",
    "adk",
    "langchain",
    "langgraph",
    "crewai",
    "generic",
  ]),
  agentArchitecture: z.enum([
    "single_agent",
    "multi_agent_coordinator",
    "retrieval_pipeline",
    "rag_rerank",
    "guarded_rag",
  ]),
  numTraces: z.number().min(100).max(5000),
  withEvals: z.boolean(),
  withDatasetAndExperiments: z.boolean(),
  scenarios: z.array(z.string()),
  toolsText: z.string().optional(),
  promptTemplateNames: z.string().optional(),
  sessionSizeMin: z.number().min(1).max(20),
  sessionSizeMax: z.number().min(1).max(50),
  promptVersionsJson: z.string().optional(),
  experimentGridModels: z.string().optional(),
});

export type DemoFormValues = z.infer<typeof demoFormSchema>;

export const defaultDemoFormValues: DemoFormValues = {
  accountName: "",
  additionalContext: "",
  industryOrUseCase: "",
  outputDir: "",
  skillFramework: "langgraph",
  agentArchitecture: "single_agent",
  numTraces: 500,
  withEvals: true,
  withDatasetAndExperiments: true,
  scenarios: [],
  toolsText: "",
  promptTemplateNames: "",
  sessionSizeMin: 3,
  sessionSizeMax: 6,
  promptVersionsJson: "",
  experimentGridModels: "",
};

/**
 * Custom hook for demo form state management.
 * Uses react-hook-form with zod validation.
 */
export function useDemoForm() {
  const form = useForm<DemoFormValues>({
    resolver: zodResolver(demoFormSchema),
    defaultValues: defaultDemoFormValues,
    mode: "onBlur",
  });

  return form;
}

/**
 * Build the API request body from form values.
 */
export function buildClassifyBody(values: DemoFormValues): Record<string, unknown> {
  const body: Record<string, unknown> = {
    account_name: values.accountName,
    industry_or_use_case: values.industryOrUseCase.trim(),
    skill_framework: values.skillFramework,
    agent_architecture: values.agentArchitecture,
    num_traces: values.numTraces,
    with_evals: values.withEvals,
    with_dataset_and_experiments: values.withDatasetAndExperiments,
  };

  if (values.additionalContext?.trim()) {
    body.additional_context = values.additionalContext.trim();
  }
  if (values.outputDir?.trim()) {
    body.output_dir = values.outputDir.trim();
  }
  if (values.scenarios.length) {
    body.scenarios = values.scenarios;
  }
  if (values.toolsText?.trim()) {
    body.tools_text = values.toolsText.trim();
  }
  if (values.promptTemplateNames?.trim()) {
    body.prompt_template_names = values.promptTemplateNames.trim();
  }
  body.session_size_min = values.sessionSizeMin;
  body.session_size_max = values.sessionSizeMax;
  if (values.promptVersionsJson?.trim()) {
    body.prompt_versions_json = values.promptVersionsJson.trim();
  }
  if (values.experimentGridModels?.trim()) {
    body.experiment_grid_models = values.experimentGridModels.trim();
  }

  return body;
}

/**
 * Framework options for the demo builder.
 */
export const SKILL_FRAMEWORK_OPTIONS = [
  { value: "openai", label: "openai" },
  { value: "anthropic", label: "anthropic" },
  { value: "bedrock", label: "bedrock" },
  { value: "vertex", label: "vertex" },
  { value: "adk", label: "adk" },
  { value: "langchain", label: "langchain" },
  { value: "langgraph", label: "langgraph" },
  { value: "crewai", label: "crewai" },
  { value: "generic", label: "generic" },
] as const;

/**
 * Agent architecture options.
 */
export const AGENT_ARCHITECTURE_OPTIONS = [
  { value: "single_agent", label: "single_agent" },
  { value: "multi_agent_coordinator", label: "multi_agent_coordinator" },
  { value: "retrieval_pipeline", label: "retrieval_pipeline" },
  { value: "rag_rerank", label: "rag_rerank" },
  { value: "guarded_rag", label: "guarded_rag" },
] as const;

/**
 * Scenario options for demo generation.
 */
export const SCENARIO_OPTIONS = [
  { value: "happy_path", label: "happy_path" },
  { value: "tool_failure", label: "tool_failure" },
  { value: "guardrail_denial", label: "guardrail_denial" },
  { value: "ambiguity", label: "ambiguity" },
  { value: "execution_failure", label: "execution_failure" },
  { value: "retry", label: "retry" },
  { value: "poisoned_tokens", label: "poisoned_tokens" },
  { value: "no_llm_needed", label: "no_llm_needed" },
] as const;

/**
 * Number of traces options.
 */
export const NUM_TRACE_OPTIONS = [100, 250, 500, 1000, 2000] as const;
