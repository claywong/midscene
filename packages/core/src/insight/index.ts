import { callAiFn } from '@/ai-model/common';
import { AiExtractElementInfo, AiLocateElement } from '@/ai-model/index';
import { AiAssert, AiLocateSection } from '@/ai-model/inspect';
import type {
  AIElementResponse,
  AISingleElementResponse,
  AIUsageInfo,
  BaseElement,
  DetailedLocateParam,
  DumpSubscriber,
  InsightAction,
  InsightAssertionResponse,
  InsightExtractParam,
  InsightOptions,
  InsightTaskInfo,
  LocateResult,
  PartialInsightDumpFromSDK,
  Rect,
  UIContext,
} from '@/types';
import {
  MIDSCENE_FORCE_DEEP_THINK,
  getAIConfigInBoolean,
  vlLocateMode,
  resetGlobalConfig,
} from '@midscene/shared/env';
import { getDebug } from '@midscene/shared/logger';
import { assert } from '@midscene/shared/utils';
import { emitInsightDump } from './utils';

export interface LocateOpts {
  callAI?: typeof callAiFn<AIElementResponse>;
  quickAnswer?: Partial<AISingleElementResponse>;
}

export type AnyValue<T> = {
  [K in keyof T]: unknown extends T[K] ? any : T[K];
};

const debug = getDebug('ai:insight');
export default class Insight<
  ElementType extends BaseElement = BaseElement,
  ContextType extends UIContext<ElementType> = UIContext<ElementType>,
> {
  contextRetrieverFn: (
    action: InsightAction,
  ) => Promise<ContextType> | ContextType;

  aiVendorFn: (...args: Array<any>) => Promise<any> = callAiFn;

  onceDumpUpdatedFn?: DumpSubscriber;

  taskInfo?: Omit<InsightTaskInfo, 'durationMs'>;

  constructor(
    context:
      | ContextType
      | ((action: InsightAction) => Promise<ContextType> | ContextType),
    opt?: InsightOptions,
  ) {
    assert(context, 'context is required for Insight');
    if (typeof context === 'function') {
      this.contextRetrieverFn = context;
    } else {
      this.contextRetrieverFn = () => Promise.resolve(context);
    }

    if (typeof opt?.aiVendorFn !== 'undefined') {
      this.aiVendorFn = opt.aiVendorFn;
    }
    if (typeof opt?.taskInfo !== 'undefined') {
      this.taskInfo = opt.taskInfo;
    }
  }

  async locate(
    query: DetailedLocateParam,
    opt?: LocateOpts,
  ): Promise<LocateResult> {
    const { callAI } = opt || {};
    const queryPrompt = typeof query === 'string' ? query : query.prompt;
    assert(
      queryPrompt || opt?.quickAnswer,
      'query or quickAnswer is required for locate',
    );
    const dumpSubscriber = this.onceDumpUpdatedFn;
    this.onceDumpUpdatedFn = undefined;

    assert(typeof query === 'object', 'query should be an object for locate');

    const globalDeepThinkSwitch = getAIConfigInBoolean(
      MIDSCENE_FORCE_DEEP_THINK,
    );
    if (globalDeepThinkSwitch) {
      debug('globalDeepThinkSwitch', globalDeepThinkSwitch);
    }
    let searchAreaPrompt;
    if (query.deepThink || globalDeepThinkSwitch || query.vlLocateMode) {
      searchAreaPrompt = query.prompt;
    }
    // 如果当前vlLocateMode为false，则使用普通模式，searchAreaPrompt设为undefined
    if (!query.vlLocateMode) {
      searchAreaPrompt = undefined;
    }

    // 新增逻辑：如果 searchAreaPrompt 存在且 vlLocateMode 不开启，且设置了VL_OPENAI相关环境变量，则临时切换到Qwen VL模式
    let needRestoreEnv = false;
    let originalEnv: Record<string, string | undefined> = {};
    if (
      searchAreaPrompt &&
      !vlLocateMode() &&
      process.env.VL_OPENAI_API_KEY &&
      process.env.VL_OPENAI_BASE_URL &&
      process.env.VL_MIDSCENE_MODEL_NAME
    ) {
      // 记录原有环境变量
      originalEnv = {
        MIDSCENE_USE_QWEN_VL: process.env.MIDSCENE_USE_QWEN_VL,
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
        MIDSCENE_MODEL_NAME: process.env.MIDSCENE_MODEL_NAME,
        MIDSCENE_MODEL_MINI_NAME: process.env.MIDSCENE_MODEL_MINI_NAME, 
      };
      // 临时切换到Qwen VL模式
      process.env.MIDSCENE_USE_QWEN_VL = '1';
      process.env.OPENAI_API_KEY = process.env.VL_OPENAI_API_KEY;
      process.env.OPENAI_BASE_URL = process.env.VL_OPENAI_BASE_URL;
      process.env.MIDSCENE_MODEL_NAME = process.env.VL_MIDSCENE_MODEL_NAME;
      delete process.env.MIDSCENE_MODEL_MINI_NAME;
      needRestoreEnv = true;
      resetGlobalConfig();
    }

    try {
      if (searchAreaPrompt && !vlLocateMode()) {
        console.warn(
          'The "deepThink" feature is not supported with multimodal LLM. Please config VL model for Midscene. https://midscenejs.com/choose-a-model',
        );
        searchAreaPrompt = undefined;
      }

      const context = await this.contextRetrieverFn('locate');

      let searchArea: Rect | undefined = undefined;
      let searchAreaRawResponse: string | undefined = undefined;
      let searchAreaUsage: AIUsageInfo | undefined = undefined;
      let searchAreaResponse:
        | Awaited<ReturnType<typeof AiLocateSection>>
        | undefined = undefined;
      if (searchAreaPrompt) {
        searchAreaResponse = await AiLocateSection({
          context,
          sectionDescription: searchAreaPrompt,
        });
        assert(
          searchAreaResponse.rect,
          `cannot find search area for "${searchAreaPrompt}"${
            searchAreaResponse.error ? `: ${searchAreaResponse.error}` : ''
          }`,
        );
        searchAreaRawResponse = searchAreaResponse.rawResponse;
        searchAreaUsage = searchAreaResponse.usage;
        searchArea = searchAreaResponse.rect;
      }

      const startTime = Date.now();
      const { parseResult, rect, elementById, rawResponse, usage } =
        await AiLocateElement({
          callAI: callAI || this.aiVendorFn,
          context,
          targetElementDescription: queryPrompt,
          quickAnswer: opt?.quickAnswer,
          searchConfig: searchAreaResponse,
        });

      const timeCost = Date.now() - startTime;
      const taskInfo: InsightTaskInfo = {
        ...(this.taskInfo ? this.taskInfo : {}),
        durationMs: timeCost,
        rawResponse: JSON.stringify(rawResponse),
        formatResponse: JSON.stringify(parseResult),
        usage,
        searchArea,
        searchAreaRawResponse,
        searchAreaUsage,
      };

      let errorLog: string | undefined;
      if (parseResult.errors?.length) {
        errorLog = `AI model failed to locate: \n${parseResult.errors.join('\n')}`;
      }

      const dumpData: PartialInsightDumpFromSDK = {
        type: 'locate',
        userQuery: {
          element: queryPrompt,
        },
        quickAnswer: opt?.quickAnswer,
        matchedElement: [],
        matchedRect: rect,
        data: null,
        taskInfo,
        deepThink: !!searchArea,
        error: errorLog,
      };

      const elements: BaseElement[] = [];
      (parseResult.elements || []).forEach((item) => {
        if ('id' in item) {
          const element = elementById(item.id);

          if (!element) {
            console.warn(
              `locate: cannot find element id=${item.id}. Maybe an unstable response from AI model`,
            );
            return;
          }
          elements.push(element);
        }
      });

      emitInsightDump(
        {
          ...dumpData,
          matchedElement: elements,
        },
        dumpSubscriber,
      );

      if (errorLog) {
        throw new Error(errorLog);
      }

      assert(
        elements.length <= 1,
        `locate: multiple elements found, length = ${elements.length}`,
      );

      if (elements.length === 1) {
        return {
          element: {
            id: elements[0]!.id,
            indexId: elements[0]!.indexId,
            center: elements[0]!.center,
            rect: elements[0]!.rect,
          },
          rect,
        };
      }
      return {
        element: null,
        rect,
      };
    } finally {
      if (needRestoreEnv) {
        if (originalEnv.MIDSCENE_USE_QWEN_VL !== undefined) {
          process.env.MIDSCENE_USE_QWEN_VL = originalEnv.MIDSCENE_USE_QWEN_VL;
        } else {
          delete process.env.MIDSCENE_USE_QWEN_VL;
        }
        if (originalEnv.OPENAI_API_KEY !== undefined) {
          process.env.OPENAI_API_KEY = originalEnv.OPENAI_API_KEY;
        } else {
          delete process.env.OPENAI_API_KEY;
        }
        if (originalEnv.OPENAI_BASE_URL !== undefined) {
          process.env.OPENAI_BASE_URL = originalEnv.OPENAI_BASE_URL;
        } else {
          delete process.env.OPENAI_BASE_URL;
        }
        if (originalEnv.MIDSCENE_MODEL_NAME !== undefined) {
          process.env.MIDSCENE_MODEL_NAME = originalEnv.MIDSCENE_MODEL_NAME;
        } else {
          delete process.env.MIDSCENE_MODEL_NAME;
        }
        if (originalEnv.MIDSCENE_MODEL_MINI_NAME !== undefined) {
          process.env.MIDSCENE_MODEL_MINI_NAME = originalEnv.MIDSCENE_MODEL_MINI_NAME;
        } else {
          delete process.env.MIDSCENE_MODEL_MINI_NAME;
        }
        resetGlobalConfig();
      }
    }
  }

  async extract<T = any>(input: string): Promise<T>;
  async extract<T extends Record<string, string>>(
    input: T,
  ): Promise<Record<keyof T, any>>;
  async extract<T extends object>(input: Record<keyof T, string>): Promise<T>;

  async extract<T>(dataDemand: InsightExtractParam): Promise<any> {
    assert(
      typeof dataDemand === 'object' || typeof dataDemand === 'string',
      `dataDemand should be object or string, but get ${typeof dataDemand}`,
    );
    const dumpSubscriber = this.onceDumpUpdatedFn;
    this.onceDumpUpdatedFn = undefined;

    const context = await this.contextRetrieverFn('extract');

    const startTime = Date.now();
    const { parseResult, usage } = await AiExtractElementInfo<T>({
      context,
      dataQuery: dataDemand,
    });

    const timeCost = Date.now() - startTime;
    const taskInfo: InsightTaskInfo = {
      ...(this.taskInfo ? this.taskInfo : {}),
      durationMs: timeCost,
      rawResponse: JSON.stringify(parseResult),
    };

    let errorLog: string | undefined;
    if (parseResult.errors?.length) {
      errorLog = `AI response error: \n${parseResult.errors.join('\n')}`;
    }

    const dumpData: PartialInsightDumpFromSDK = {
      type: 'extract',
      userQuery: {
        dataDemand,
      },
      matchedElement: [],
      data: null,
      taskInfo,
      error: errorLog,
    };

    const { data } = parseResult || {};

    // 4
    emitInsightDump(
      {
        ...dumpData,
        data,
      },
      dumpSubscriber,
    );

    if (errorLog && !data) {
      throw new Error(errorLog);
    }

    return {
      data,
      usage,
    };
  }

  async assert(assertion: string): Promise<InsightAssertionResponse> {
    if (typeof assertion !== 'string') {
      throw new Error(
        'This is the assert method for Midscene, the first argument should be a string. If you want to use the assert method from Node.js, please import it from the Node.js assert module.',
      );
    }

    const dumpSubscriber = this.onceDumpUpdatedFn;
    this.onceDumpUpdatedFn = undefined;

    const context = await this.contextRetrieverFn('assert');
    const startTime = Date.now();
    const assertResult = await AiAssert({
      assertion,
      context,
    });

    const timeCost = Date.now() - startTime;
    const taskInfo: InsightTaskInfo = {
      ...(this.taskInfo ? this.taskInfo : {}),
      durationMs: timeCost,
      rawResponse: JSON.stringify(assertResult.content),
    };

    const { thought, pass } = assertResult.content;
    const dumpData: PartialInsightDumpFromSDK = {
      type: 'assert',
      userQuery: {
        assertion,
      },
      matchedElement: [],
      data: null,
      taskInfo,
      assertionPass: pass,
      assertionThought: thought,
      error: pass ? undefined : thought,
    };
    emitInsightDump(dumpData, dumpSubscriber);

    return {
      pass,
      thought,
      usage: assertResult.usage,
    };
  }
}
