import {Reporter, ReporterOnStartOptions, Test, TestCaseResult, TestContext} from "@jest/reporters";
import {AggregatedResult} from "@jest/test-result";

type CustomReporter = Pick<Reporter, "onRunComplete"|"onTestResult">;

export default class JestSlowTestReporter implements CustomReporter
{
   _globalConfig: any;
   _options: any;
   _slowTests: any[];
   constructor(globalConfig: any, options: any)
   {
      this._globalConfig = globalConfig;
      this._options      = options;
      this._slowTests    = [];
   }

   onRunComplete()
   {
      console.log();
      this._slowTests.sort((a: {duration: number;}, b: {duration: number;}) => b.duration - a.duration);
      var rootPathRegex = new RegExp(`^${process.cwd()}`);
      var slowestTests  = this._slowTests.slice(0, this._options.numTests || 10);
      var slowTestTime  = this._slowTestTime(slowestTests);
      var allTestTime   = this._allTestTime();
      var percentTime   = (slowTestTime / allTestTime) * 100;

      console.log(`Top ${slowestTests.length} slowest examples (${slowTestTime / 1000} seconds,` +
                  ` ${percentTime.toFixed(1)}% of total time):`);

      for (var i = 0; i < slowestTests.length; i++)
      {
         var duration = slowestTests[i].duration;
         var fullName = slowestTests[i].fullName;
         var filePath = slowestTests[i].filePath.replace(rootPathRegex, ".");

         console.log(`  ${fullName}`);
         console.log(`    ${duration / 1000} seconds ${filePath}`);
      }
      console.log();
   }

   onTestResult(test: any, testResult: {testResults: string|any[]; testFilePath: any;})
   {
      for (var i = 0; i < testResult.testResults.length; i++)
      {
         this._slowTests.push({
            duration: testResult.testResults[i].duration,
            fullName: testResult.testResults[i].fullName,
            filePath: testResult.testFilePath,
         });
      }
   }

   onTestCaseResult(test: Test, testCaseResult: TestCaseResult)
   {
      console.log("Test case result");
   }

   onRunStart(results: AggregatedResult, options: ReporterOnStartOptions)
   {
      console.log("Test case result");
   }
   onTestStart(test: Test)
   {
      console.log("Test case result");
   }
   onTestFileStart(test: Test)
   {
      console.log("Test case result");
   }

   _slowTestTime(slowestTests: string|any[])
   {
      var slowTestTime = 0;
      for (var i = 0; i < slowestTests.length; i++)
      {
         slowTestTime += slowestTests[i].duration;
      }
      return slowTestTime;
   }

   _allTestTime()
   {
      var allTestTime = 0;
      for (var i = 0; i < this._slowTests.length; i++)
      {
         allTestTime += this._slowTests[i].duration;
      }
      return allTestTime;
   }
}
